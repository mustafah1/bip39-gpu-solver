use std::fs;
use std::ffi::CString;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::env;
use std::process::Command;
use ocl::{core, flags};
use ocl::enums::ArgVal;
use ocl::builders::ContextProperties;
use ocl::core::Error as OclCoreError;
// use std::time::Instant; // Unused
use std::io::{Write}; // stderr unused

// Our 12 words - BIP39 strings
const TOTAL_PERMS: u64 = 479_001_600;
const INITIAL_BATCH: usize = 64;
const BATCH_CAP: usize = 16384;
const LOCAL_WORK_SIZES: [usize; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];
const BUILD_HEARTBEAT_SECS: u64 = 10;
const THROUGHPUT_REPORT_SECS: u64 = 5;
const BATCH_GROW_ITERS: u32 = 100;
const READ_BACK_EVERY: u32 = 8;

const WORD_STRINGS: [&str; 12] = [
    "asset", "basket", "capital", "execute", "gauge", "improve",
    "pair", "price", "require", "sell", "share", "trend"
];

// Use stderr for debug since it's unbuffered
macro_rules! dbg_print {
    ($($arg:tt)*) => {{
        eprintln!($($arg)*);
    }};
}

fn factorial(n: u64) -> u64 {
    match n { 0 | 1 => 1, _ => (2..=n).product() }
}

fn permutation_to_indices(mut k: u64) -> [usize; 12] {
    let mut indices: Vec<usize> = (0..12).collect();
    let mut result = [0usize; 12];
    for i in (1..=12).rev() {
        let f = factorial((i - 1) as u64);
        let j = (k / f) as usize;
        k = k % f;
        result[12 - i] = indices.remove(j);
    }
    result
}

fn perm_to_words(perm: &[usize; 12]) -> String {
    perm.iter().map(|&i| WORD_STRINGS[i]).collect::<Vec<_>>().join(" ")
}

fn is_out_of_resources(err: &OclCoreError) -> bool {
    format!("{:?}", err).contains("CL_OUT_OF_RESOURCES")
}

fn parse_gpu_stats_interval() -> Option<u64> {
    for arg in env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--gpu-stats=") {
            if let Ok(secs) = val.parse::<u64>() {
                return Some(secs.max(1));
            }
        } else if arg == "--gpu-stats" {
            return Some(5);
        }
    }
    None
}

fn parse_shard_args() -> (u32, u32) {
    let mut shard_count: u32 = 1;
    let mut shard_index: u32 = 0;
    for arg in env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--shard-count=") {
            if let Ok(count) = val.parse::<u32>() {
                shard_count = count.max(1);
            }
        } else if let Some(val) = arg.strip_prefix("--shard-index=") {
            if let Ok(index) = val.parse::<u32>() {
                shard_index = index;
            }
        }
    }
    if shard_index >= shard_count {
        eprintln!("[WARN] shard-index {} >= shard-count {}; clamping to 0", shard_index, shard_count);
        shard_index = 0;
    }
    (shard_count, shard_index)
}

fn parse_device_index() -> usize {
    for arg in env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--device-index=") {
            if let Ok(idx) = val.parse::<usize>() {
                return idx;
            }
        }
    }
    0
}

fn parse_range_args() -> (u64, u64) {
    let mut start: u64 = 0;
    let mut end: u64 = TOTAL_PERMS;
    for arg in env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--start=") {
            if let Ok(v) = val.parse::<u64>() {
                start = v.min(TOTAL_PERMS);
            }
        } else if let Some(val) = arg.strip_prefix("--end=") {
            if let Ok(v) = val.parse::<u64>() {
                end = v.min(TOTAL_PERMS);
            }
        }
    }
    if end < start {
        eprintln!("[WARN] end < start, swapping values");
        (end, start)
    } else {
        (start, end)
    }
}

fn start_gpu_stats_thread(interval_secs: u64, stop: Arc<AtomicBool>) {
    thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            let output = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ])
                .output();
            if let Ok(out) = output {
                if out.status.success() {
                    if let Ok(text) = String::from_utf8(out.stdout) {
                        let line = text.trim();
                        if !line.is_empty() {
                            eprintln!("[GPU] util {}, mem util {}, {} MiB / {} MiB", 
                                line.split(',').nth(0).unwrap_or("?").trim(),
                                line.split(',').nth(1).unwrap_or("?").trim(),
                                line.split(',').nth(2).unwrap_or("?").trim(),
                                line.split(',').nth(3).unwrap_or("?").trim(),
                            );
                        }
                    }
                }
            }
            thread::sleep(Duration::from_secs(interval_secs));
        }
    });
}

fn load_kernel_source() -> String {
    // Exclude 'secp256k1_prec' because we load it manually into a buffer
    let files = ["common", "ripemd", "sha2", "secp256k1_common", "secp256k1_scalar", 
                 "secp256k1_field", "secp256k1_group", "secp256k1", 
                 "address", "mnemonic_constants", "int_to_address"];
    files.iter()
        .map(|f| fs::read_to_string(format!("./cl/{}.cl", f)).expect(&format!("Failed: {}", f)))
        .collect::<Vec<_>>()
        .join("\n")
}

// Parse secp256k1_prec.cl to extract the flat list of u32s
fn load_prec_table() -> Vec<u32> {
    dbg_print!("[DBG] Parsing secp256k1_prec.cl table...");
    let content = fs::read_to_string("./cl/secp256k1_prec.cl").expect("Failed to read prec table");
    let mut values = Vec::new();

    // Split by "SC("
    let parts: Vec<&str> = content.split("SC(").collect();
    
    // Skip the first part (header)
    for part in parts.iter().skip(1) {
        // Take the content until ')'
        let end_idx = part.find(')').unwrap_or(part.len());
        let sc_content = &part[0..end_idx];
        
        let nums: Vec<&str> = sc_content.split(',').collect();
        for num_str in nums {
             let clean_str: String = num_str.chars().filter(|c| c.is_digit(10)).collect();
             if let Ok(num) = clean_str.parse::<u32>() {
                 values.push(num);
             }
        }
    }
    
    dbg_print!("[DBG] Parsed {} u32 values (Expected: 128*4*16 = {})", values.len(), 128*4*16);
    if values.len() != 8192 {
        panic!("Parsed wrong number of values from prec table!");
    }
    
    values
}

fn main() {
    dbg_print!("[DBG] Starting...");

    let gpu_stats_interval = parse_gpu_stats_interval();
    let gpu_stats_stop = Arc::new(AtomicBool::new(false));
    if let Some(secs) = gpu_stats_interval {
        eprintln!("[DBG] GPU stats enabled (every {}s)", secs);
        start_gpu_stats_thread(secs, Arc::clone(&gpu_stats_stop));
    }

    let (shard_count, shard_index) = parse_shard_args();
    if shard_count > 1 {
        eprintln!("[DBG] Sharding enabled: {}/{}", shard_index, shard_count);
    }
    let (range_start, range_end) = parse_range_args();
    if range_start != 0 || range_end != TOTAL_PERMS {
        eprintln!("[DBG] Range enabled: {} -> {}", range_start, range_end);
    }
    
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     GPU BIP39 12-Word Permutation Scanner                  ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Target: 3CKkHm2nTS46vrTiGayj4fPtggjq8opcZF                  ║");
    println!("║ Words:  asset basket capital execute gauge improve         ║");
    println!("║         pair price require sell share trend                ║");
    println!("║ Total:  479,001,600 permutations                           ║");
    println!("║ Batch:  {} GPU work items/call                           ║", INITIAL_BATCH);
    println!("╚════════════════════════════════════════════════════════════╝");
    
    // 1. Load Data First
    let prec_data = load_prec_table();

    dbg_print!("[DBG] Getting platform...");
    let platform_id = core::default_platform().expect("No OpenCL platform");
    
    dbg_print!("[DBG] Getting devices...");
    let device_ids = core::get_device_ids(&platform_id, Some(ocl::flags::DEVICE_TYPE_GPU), None)
        .expect("No GPU");
    
    println!("\n✅ Found {} GPU(s)", device_ids.len());
    let device_index = parse_device_index();
    if device_index >= device_ids.len() {
        eprintln!("[WARN] device-index {} out of range, defaulting to 0", device_index);
    }
    let device_id = device_ids[device_index.min(device_ids.len().saturating_sub(1))];
    let dev_name = core::get_device_info(&device_id, core::DeviceInfo::Name).unwrap();
    println!("✅ Using: {}", dev_name);
    
    dbg_print!("[DBG] Creating context...");
    let context_properties = ContextProperties::new().platform(platform_id);
    let context = core::create_context(Some(&context_properties), &[device_id], None, None).unwrap();
    
    dbg_print!("[DBG] Loading kernel source...");
    println!("\n🔧 Compiling OpenCL kernels (Lite Mode)...");
    let src = CString::new(load_kernel_source()).unwrap();
    
    dbg_print!("[DBG] Creating program...");
    let program = core::create_program_with_source(&context, &[src]).unwrap();
    
    dbg_print!("[DBG] Building program...");
    let build_done = Arc::new(AtomicBool::new(false));
    let build_done_thread = Arc::clone(&build_done);
    let build_start = Instant::now();
    let heartbeat = thread::spawn(move || {
        while !build_done_thread.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(BUILD_HEARTBEAT_SECS));
            if build_done_thread.load(Ordering::Relaxed) {
                break;
            }
            let elapsed = build_start.elapsed().as_secs();
            eprintln!("[DBG] Kernel build still running... {}s", elapsed);
        }
    });

    // Enable math optimizations to reduce register pressure and avoid resource exhaustion.
    if let Err(e) = core::build_program(&program, Some(&[device_id]), &CString::new("-cl-mad-enable").unwrap(), None, None) {
        build_done.store(true, Ordering::Relaxed);
        let _ = heartbeat.join();
        eprintln!("Kernel build error: {:?}", e);
        return;
    }
    build_done.store(true, Ordering::Relaxed);
    let _ = heartbeat.join();
    
    dbg_print!("[DBG] Program built successfully!");
    println!("✅ Kernels compiled");
    
    dbg_print!("[DBG] Creating command queue...");
    let mut queue = core::create_command_queue(&context, &device_id, None).unwrap();
    
    dbg_print!("[DBG] Creating kernel...");
    let kernel = core::create_kernel(&program, "int_to_address").unwrap();
    
    // Buffers
    dbg_print!("[DBG] Allocating host arrays...");
    let target_mnemonic = vec![0u8; 180];
    let mut found_result = vec![0u8; 8];
    
    dbg_print!("[DBG] Creating GPU buffers...");
    let (found_buf, target_buf, prec_buf) = unsafe {
        let fb = core::create_buffer(&context, flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR, found_result.len(), Some(&found_result)).unwrap();
        let tb = core::create_buffer(&context, flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR, 180, Some(&target_mnemonic)).unwrap();
        let pb = core::create_buffer(&context, flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR, prec_data.len(), Some(&prec_data)).unwrap();
        (fb, tb, pb)
    };

    dbg_print!("[DBG] All setup complete!");
    println!("✅ Ready\n");
    println!("🚀 Starting GPU search...");

    let mut k: u64 = range_start + shard_index as u64;
    let mut max_batch = INITIAL_BATCH.min(BATCH_CAP);
    let mut local_work_size = LOCAL_WORK_SIZES.iter().copied().find(|&s| s <= max_batch).unwrap_or(1);
    let mut last_report = Instant::now();
    let mut last_k: u64 = 0;
    let mut success_iters: u32 = 0;
    let mut read_counter: u32 = 0;

    // Kernel args that don't change each iteration
    core::set_kernel_arg(&kernel, 2, ArgVal::mem(&target_buf)).unwrap();
    core::set_kernel_arg(&kernel, 3, ArgVal::mem(&found_buf)).unwrap();
    core::set_kernel_arg(&kernel, 4, ArgVal::mem(&prec_buf)).unwrap();
    
    while k < range_end {
        if local_work_size > max_batch {
            local_work_size = LOCAL_WORK_SIZES.iter().copied().find(|&s| s <= max_batch).unwrap_or(1);
        }
        let remaining = range_end.saturating_sub(k);
        let max_batch_u64 = max_batch as u64;
        let stride_u64 = shard_count as u64;
        let max_items = if remaining == 0 {
            0
        } else {
            ((remaining - 1) / stride_u64) + 1
        };
        let actual_batch = (max_items.min(max_batch_u64)) as usize;
        if actual_batch == 0 {
            break;
        }

        if read_counter == 0 {
            found_result.fill(0);
            let write_found = unsafe {
                 core::enqueue_write_buffer(&queue, &found_buf, true, 0, &found_result, None::<&core::Event>, None::<&mut core::Event>)
            };
            if let Err(e) = write_found {
                if is_out_of_resources(&e) {
                    if max_batch > 1 {
                        max_batch = (max_batch / 2).max(1);
                        eprintln!("[DBG] CL_OUT_OF_RESOURCES on write found; reducing batch to {}", max_batch);
                    } else {
                        eprintln!("[DBG] CL_OUT_OF_RESOURCES on write found at batch 1; retrying with fresh queue");
                    }
                    queue = core::create_command_queue(&context, &device_id, None).unwrap();
                    success_iters = 0;
                    continue;
                }
                panic!("Write buffer (found) error: {:?}", e);
            }
        }
        
        // Arguments: 0=start_k, 1=stride, 2=target, 3=found, 4=prec_table, 5=batch_len
        let stride = shard_count;
        core::set_kernel_arg(&kernel, 0, ArgVal::scalar(&k)).unwrap();
        core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&stride)).unwrap();
        core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&(actual_batch as u32))).unwrap();

        // Run
        let padded_batch = ((actual_batch + local_work_size - 1) / local_work_size) * local_work_size;
        let global_work_size = [padded_batch, 1, 1];
        let local_work_size_arr = [local_work_size, 1, 1];
        let enqueue_res = unsafe {
            core::enqueue_kernel(&queue, &kernel, 1, None, &global_work_size, Some(local_work_size_arr), None::<&core::Event>, None::<&mut core::Event>)
        };
        if let Err(e) = enqueue_res {
            if is_out_of_resources(&e) {
                if max_batch > 1 {
                    max_batch = (max_batch / 2).max(1);
                    eprintln!("[DBG] CL_OUT_OF_RESOURCES on enqueue; reducing batch to {}", max_batch);
                } else if local_work_size > 1 {
                    local_work_size = LOCAL_WORK_SIZES.iter().copied().find(|&s| s < local_work_size).unwrap_or(1);
                    eprintln!("[DBG] CL_OUT_OF_RESOURCES on enqueue; reducing local size to {}", local_work_size);
                } else {
                    eprintln!("[DBG] CL_OUT_OF_RESOURCES on enqueue at batch 1; retrying with fresh queue");
                }
                queue = core::create_command_queue(&context, &device_id, None).unwrap();
                success_iters = 0;
                continue;
            }
            panic!("Kernel enqueue error: {:?}", e);
        }
        
         // Read result
        let should_read = read_counter + 1 >= READ_BACK_EVERY || k + (actual_batch as u64) >= TOTAL_PERMS;
        if should_read {
            let read_res = unsafe {
                core::enqueue_read_buffer(&queue, &found_buf, true, 0, &mut found_result, None::<&core::Event>, None::<&mut core::Event>)
            };
            if let Err(e) = read_res {
                if is_out_of_resources(&e) {
                    if max_batch > 1 {
                        max_batch = (max_batch / 2).max(1);
                        eprintln!("[DBG] CL_OUT_OF_RESOURCES on read; reducing batch to {}", max_batch);
                    } else if local_work_size > 1 {
                        local_work_size = LOCAL_WORK_SIZES.iter().copied().find(|&s| s < local_work_size).unwrap_or(1);
                        eprintln!("[DBG] CL_OUT_OF_RESOURCES on read; reducing local size to {}", local_work_size);
                    } else {
                        eprintln!("[DBG] CL_OUT_OF_RESOURCES on read at batch 1; retrying with fresh queue");
                    }
                    queue = core::create_command_queue(&context, &device_id, None).unwrap();
                    success_iters = 0;
                    read_counter = 0;
                    continue;
                }
                panic!("Read buffer error: {:?}", e);
            }
            read_counter = 0;
        } else {
            read_counter += 1;
        }

        if found_result[0] == 1 {
             println!("\n🎉 FOUND IT!");
             // Parse found absolute index
             let found_idx_val = ((found_result[1] as u64) << 24) |
                                 ((found_result[2] as u64) << 16) |
                                 ((found_result[3] as u64) << 8) |
                                  (found_result[4] as u64);
                                  
             println!("Match at offset: {}", found_idx_val);
             let final_scan_idx = found_idx_val;
             let indices = permutation_to_indices(final_scan_idx);
             let words = perm_to_words(&indices);
             println!("Mnemonic: {}", words);
             break;
        }

        // Progress
        if k % 100000 == 0 {
             print!("\rChecked: {} / {} ({}%)", k, TOTAL_PERMS, (k * 100 / TOTAL_PERMS));
             std::io::stdout().flush().unwrap();
        }

        if last_report.elapsed() >= Duration::from_secs(THROUGHPUT_REPORT_SECS) {
            let elapsed = last_report.elapsed().as_secs_f64().max(0.001);
            let delta = k - last_k;
            let rate = (delta as f64) / elapsed;
            let remaining = range_end.saturating_sub(k);
            let eta_secs = if rate > 0.0 { (remaining as f64) / rate } else { 0.0 };
            eprintln!(
                "[DBG] Rate: {:.0} perms/s | batch {} | local {} | eta {:.1}h",
                rate,
                max_batch,
                local_work_size,
                eta_secs / 3600.0
            );
            last_report = Instant::now();
            last_k = k;
        }

        success_iters += 1;
        if success_iters >= BATCH_GROW_ITERS && max_batch < BATCH_CAP {
            max_batch = (max_batch * 2).min(BATCH_CAP);
            local_work_size = LOCAL_WORK_SIZES.iter().copied().find(|&s| s <= max_batch).unwrap_or(1);
            eprintln!("[DBG] Increasing batch to {} (local {})", max_batch, local_work_size);
            success_iters = 0;
        }
        
        k += (actual_batch as u64) * (shard_count as u64);
    }
    
    println!("\nDone.");

    gpu_stats_stop.store(true, Ordering::Relaxed);
}
