use std::fs;
use std::ffi::CString;
use ocl::{core, flags};
use ocl::enums::ArgVal;
use ocl::builders::ContextProperties;
// use std::time::Instant; // Unused
use std::io::{Write}; // stderr unused

// Our 12 words - BIP39 indices
const WORDS: [u16; 12] = [112, 146, 238, 608, 759, 905, 1251, 1348, 1437, 1559, 1597, 1841];
const TOTAL_PERMS: u64 = 479_001_600;
const BATCH_SIZE: usize = 64; // Restored to 64 with optimizations enabled

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
    
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     GPU BIP39 12-Word Permutation Scanner                  ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Target: 3CKkHm2nTS46vrTiGayj4fPtggjq8opcZF                  ║");
    println!("║ Words:  asset basket capital execute gauge improve         ║");
    println!("║         pair price require sell share trend                ║");
    println!("║ Total:  479,001,600 permutations                           ║");
    println!("║ Batch:  {} GPU work items/call                           ║", BATCH_SIZE);
    println!("╚════════════════════════════════════════════════════════════╝");
    
    // 1. Load Data First
    let prec_data = load_prec_table();

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
    println!("\n🔧 Compiling OpenCL kernels (Lite Mode)...");
    let src = CString::new(load_kernel_source()).unwrap();
    
    dbg_print!("[DBG] Creating program...");
    let program = core::create_program_with_source(&context, &[src]).unwrap();
    
    dbg_print!("[DBG] Building program...");
    // We re-enable optimizations (-cl-mad-enable) because the kernel is now small enough!
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
    
    // Buffers
    dbg_print!("[DBG] Allocating host arrays...");
    let mut hi_arr = vec![0u64; BATCH_SIZE];
    let mut lo_arr = vec![0u64; BATCH_SIZE];
    let target_mnemonic = vec![0u8; 180];
    let mut found_result = vec![0u8; 8];
    
    dbg_print!("[DBG] Creating GPU buffers...");
    let (hi_buf, lo_buf, found_buf, target_buf, prec_buf) = unsafe {
        let hb = core::create_buffer(&context, flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR, BATCH_SIZE, Some(&hi_arr)).unwrap();
        let lb = core::create_buffer(&context, flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR, BATCH_SIZE, Some(&lo_arr)).unwrap();
        let fb = core::create_buffer(&context, flags::MEM_WRITE_ONLY | flags::MEM_COPY_HOST_PTR, 8, Some(&found_result)).unwrap();
        let tb = core::create_buffer(&context, flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR, 180, Some(&target_mnemonic)).unwrap();
        let pb = core::create_buffer(&context, flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR, prec_data.len(), Some(&prec_data)).unwrap();
        (hb, lb, fb, tb, pb)
    };

    dbg_print!("[DBG] All setup complete!");
    println!("✅ Ready\n");
    println!("🚀 Starting GPU search...");

    let mut k: u64 = 0;
    
    while k < TOTAL_PERMS {
        let actual_batch = if k + (BATCH_SIZE as u64) > TOTAL_PERMS {
            (TOTAL_PERMS - k) as usize
        } else {
            BATCH_SIZE
        };

        // Prepare batch
        for i in 0..actual_batch {
             let indices = permutation_to_indices(k + (i as u64));
             let (hi, lo) = encode_mnemonic(&indices);
             hi_arr[i] = hi;
             lo_arr[i] = lo;
        }

        // Write inputs - use slices for actual batch size
        unsafe {
             core::enqueue_write_buffer(&queue, &hi_buf, true, 0, &hi_arr[0..actual_batch], None::<&core::Event>, None::<&mut core::Event>).unwrap();
             core::enqueue_write_buffer(&queue, &lo_buf, true, 0, &lo_arr[0..actual_batch], None::<&core::Event>, None::<&mut core::Event>).unwrap();
        }
        
        // Arguments: 0=hi, 1=lo, 2=target, 3=found, 4=prec_table
        core::set_kernel_arg(&kernel, 0, ArgVal::mem(&hi_buf)).unwrap();
        core::set_kernel_arg(&kernel, 1, ArgVal::mem(&lo_buf)).unwrap();
        core::set_kernel_arg(&kernel, 2, ArgVal::mem(&target_buf)).unwrap();
        core::set_kernel_arg(&kernel, 3, ArgVal::mem(&found_buf)).unwrap();
        core::set_kernel_arg(&kernel, 4, ArgVal::mem(&prec_buf)).unwrap();

        // Run
        let global_work_size = [actual_batch, 1, 1];
        unsafe {
            core::enqueue_kernel(&queue, &kernel, 1, None, &global_work_size, None, None::<&core::Event>, None::<&mut core::Event>).unwrap();
        }
        
         // Read result
        unsafe {
            core::enqueue_read_buffer(&queue, &found_buf, true, 0, &mut found_result, None::<&core::Event>, None::<&mut core::Event>).unwrap();
        }

        if found_result[0] == 1 {
             println!("\n🎉 FOUND IT!");
             // Parse found index
             let found_idx_val = ((found_result[1] as u64) << 24) | 
                                 ((found_result[2] as u64) << 16) | 
                                 ((found_result[3] as u64) << 8) | 
                                  (found_result[4] as u64);
                                  
             println!("Match at offset: {}", found_idx_val);
             let final_scan_idx = k + found_idx_val;
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
        
        k += actual_batch as u64;
    }
    
    println!("\nDone.");
}
