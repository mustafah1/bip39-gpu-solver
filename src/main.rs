use std::fs;
use std::ffi::CString;
use ocl::{core, flags};
use ocl::enums::ArgVal;
use ocl::builders::ContextProperties;
use std::time::Instant;
use std::io::{Write, stderr};

// Our 12 words - BIP39 indices
const WORDS: [u16; 12] = [112, 146, 238, 608, 759, 905, 1251, 1348, 1437, 1559, 1597, 1841];
const TOTAL_PERMS: u64 = 479_001_600;
const BATCH_SIZE: usize = 1; // Testing absolute minimum to diagnose CL_OUT_OF_RESOURCES



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

fn encode_mnemonic(perm_indices: &[usize; 12]) -> (u64, u64) {
    let mut bits: u128 = 0;
    for i in 0..12 {
        bits = (bits << 11) | (WORDS[perm_indices[i]] as u128);
    }
    bits <<= 4;
    ((bits >> 64) as u64, bits as u64)
}

fn perm_to_words(perm: &[usize; 12]) -> String {
    perm.iter().map(|&i| WORD_STRINGS[i]).collect::<Vec<_>>().join(" ")
}

fn load_kernel_source() -> String {
    let files = ["common", "ripemd", "sha2", "secp256k1_common", "secp256k1_scalar", 
                 "secp256k1_field", "secp256k1_group", "secp256k1_prec", "secp256k1", 
                 "address", "mnemonic_constants", "int_to_address"];
    files.iter()
        .map(|f| fs::read_to_string(format!("./cl/{}.cl", f)).expect(&format!("Failed: {}", f)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    dbg_print!("[DBG] Starting...");
    
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     GPU BIP39 12-Word Permutation Scanner                  ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Target: 3CKkHm2nTS46vrTiGayj4fPtggjq8opcZF                  ║");
    println!("║ Words:  asset basket capital execute gauge improve         ║");
    println!("║         pair price require sell share trend                ║");
    println!("║ Total:  479,001,600 permutations                           ║");
    println!("║ Batch:  {} GPU work items/call                           ║", BATCH_SIZE);
    println!("╚════════════════════════════════════════════════════════════╝");
    
    dbg_print!("[DBG] Getting platform...");
    let platform_id = core::default_platform().expect("No OpenCL platform");
    
    dbg_print!("[DBG] Getting devices...");
    let device_ids = core::get_device_ids(&platform_id, Some(ocl::flags::DEVICE_TYPE_GPU), None)
        .expect("No GPU");
    
    println!("\n✅ Found {} GPU(s)", device_ids.len());
    let device_id = device_ids[0];
    let dev_name = core::get_device_info(&device_id, core::DeviceInfo::Name).unwrap();
    println!("✅ Using: {}", dev_name);
    
    dbg_print!("[DBG] Creating context...");
    let context_properties = ContextProperties::new().platform(platform_id);
    let context = core::create_context(Some(&context_properties), &[device_id], None, None).unwrap();
    
    dbg_print!("[DBG] Loading kernel source...");
    println!("\n🔧 Compiling OpenCL kernels...");
    let src = CString::new(load_kernel_source()).unwrap();
    
    dbg_print!("[DBG] Creating program...");
    let program = core::create_program_with_source(&context, &[src]).unwrap();
    
    dbg_print!("[DBG] Building program...");
    if let Err(e) = core::build_program(&program, Some(&[device_id]), &CString::new("-cl-opt-disable").unwrap(), None, None) {
        eprintln!("Kernel build error: {:?}", e);
        return;
    }
    
    dbg_print!("[DBG] Program built successfully!");
    println!("✅ Kernels compiled");
    
    dbg_print!("[DBG] Creating command queue...");
    let queue = core::create_command_queue(&context, &device_id, None).unwrap();
    
    dbg_print!("[DBG] Creating kernel...");
    let kernel = core::create_kernel(&program, "int_to_address").unwrap();
    
    dbg_print!("[DBG] Allocating host arrays...");
    let mut hi_arr = vec![0u64; BATCH_SIZE];
    let mut lo_arr = vec![0u64; BATCH_SIZE];
    let mut target_mnemonic = vec![0u8; 180];
    let mut found_result = vec![0u8; 8];

    dbg_print!("[DBG] Creating GPU buffers...");
    let hi_buf = unsafe { 
        core::create_buffer(&context, flags::MEM_READ_ONLY, BATCH_SIZE * 8, None::<&[u64]>).unwrap()
    };
    let lo_buf = unsafe { 
        core::create_buffer(&context, flags::MEM_READ_ONLY, BATCH_SIZE * 8, None::<&[u64]>).unwrap()
    };
    let target_buf = unsafe { 
        core::create_buffer(&context, flags::MEM_WRITE_ONLY, 180, None::<&[u8]>).unwrap()
    };
    let found_buf = unsafe {
        core::create_buffer(&context, flags::MEM_READ_WRITE, 8, None::<&[u8]>).unwrap()
    };
    
    dbg_print!("[DBG] All setup complete!");
    println!("✅ Ready\n");
    println!("🚀 Starting GPU search...\n");

    let start_time = Instant::now();
    let mut checked: u64 = 0;
    let mut perm_k: u64 = 0;
    
    while perm_k < TOTAL_PERMS {
        let batch_end = std::cmp::min(perm_k + BATCH_SIZE as u64, TOTAL_PERMS);
        let actual_batch = (batch_end - perm_k) as usize;
        
        // Encode permutations
        for i in 0..actual_batch {
            let perm = permutation_to_indices(perm_k + i as u64);
            let (hi, lo) = encode_mnemonic(&perm);
            hi_arr[i] = hi;
            lo_arr[i] = lo;
        }
        
        found_result[0] = 0;
        
        // Write to GPU
        unsafe {
            core::enqueue_write_buffer(&queue, &hi_buf, false, 0, &hi_arr[..actual_batch], 
                None::<core::Event>, None::<&mut core::Event>).unwrap();
            core::enqueue_write_buffer(&queue, &lo_buf, false, 0, &lo_arr[..actual_batch], 
                None::<core::Event>, None::<&mut core::Event>).unwrap();
            core::enqueue_write_buffer(&queue, &found_buf, true, 0, &found_result, 
                None::<core::Event>, None::<&mut core::Event>).unwrap();
        }
        
        // Set kernel args
        core::set_kernel_arg(&kernel, 0, ArgVal::mem(&hi_buf)).unwrap();
        core::set_kernel_arg(&kernel, 1, ArgVal::mem(&lo_buf)).unwrap();
        core::set_kernel_arg(&kernel, 2, ArgVal::mem(&target_buf)).unwrap();
        core::set_kernel_arg(&kernel, 3, ArgVal::mem(&found_buf)).unwrap();
        
        // Execute
        unsafe {
            core::enqueue_kernel(&queue, &kernel, 1, None, &[actual_batch, 1, 1], 
                None, None::<core::Event>, None::<&mut core::Event>).unwrap();
        }
        
        // Read result
        unsafe {
            core::enqueue_read_buffer(&queue, &found_buf, true, 0, &mut found_result, 
                None::<core::Event>, None::<&mut core::Event>).unwrap();
        }
        
        if found_result[0] == 0x01 {
            unsafe {
                core::enqueue_read_buffer(&queue, &target_buf, true, 0, &mut target_mnemonic, 
                    None::<core::Event>, None::<&mut core::Event>).unwrap();
            }
            let found_idx = ((found_result[1] as u32) << 24) | ((found_result[2] as u32) << 16) 
                          | ((found_result[3] as u32) << 8) | (found_result[4] as u32);
            let perm_num = perm_k + found_idx as u64;
            let perm = permutation_to_indices(perm_num);
            let mnemonic = String::from_utf8_lossy(&target_mnemonic)
                .trim_matches(char::from(0)).to_string();
            
            println!("\n🎉 FOUND!");
            println!("Mnemonic: {}", mnemonic);
            println!("Words: {}", perm_to_words(&perm));
            println!("Permutation: {}", perm_num);
            println!("Time: {:.2}s", start_time.elapsed().as_secs_f64());
            
            if let Ok(mut file) = fs::File::create("/content/FOUND.txt") {
                writeln!(file, "FOUND: {}", mnemonic).unwrap();
            }
            return;
        }
        
        checked += actual_batch as u64;
        perm_k = batch_end;
        
        if checked % 25600 < BATCH_SIZE as u64 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = checked as f64 / elapsed;
            let eta = (TOTAL_PERMS - checked) as f64 / rate;
            let pct = (checked as f64 / TOTAL_PERMS as f64) * 100.0;
            eprint!("\r[{:.2}%] {}/{} | {:.0}/s | ETA: {:.1}m    ", 
                pct, checked, TOTAL_PERMS, rate, eta / 60.0);
        }
    }
    
    println!("\n❌ Not found");
}
