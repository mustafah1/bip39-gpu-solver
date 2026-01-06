use std::fs;
use std::ffi::CString;
use ocl::{core, flags};
use ocl::enums::ArgVal;
use ocl::builders::ContextProperties;
use std::time::Instant;
use std::io::Write;

// Our 12 words - BIP39 indices (sorted alphabetically)
// asset=112, basket=146, capital=238, execute=608, gauge=759, improve=905
// pair=1251, price=1348, require=1437, sell=1559, share=1597, trend=1841
const WORDS: [u16; 12] = [112, 146, 238, 608, 759, 905, 1251, 1348, 1437, 1559, 1597, 1841];
const TOTAL_PERMS: u64 = 479_001_600;

// BIP39 word list for output
const WORD_STRINGS: [&str; 12] = [
    "asset", "basket", "capital", "execute", "gauge", "improve",
    "pair", "price", "require", "sell", "share", "trend"
];

fn factorial(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        _ => (2..=n).product()
    }
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
    bits <<= 4; // leave 4 bits for checksum (calculated by GPU)
    ((bits >> 64) as u64, bits as u64)
}

fn perm_to_words(perm: &[usize; 12]) -> String {
    perm.iter()
        .map(|&i| WORD_STRINGS[i])
        .collect::<Vec<_>>()
        .join(" ")
}

fn load_kernel_source() -> String {
    let files = ["common", "ripemd", "sha2", "secp256k1_common", "secp256k1_scalar", 
                 "secp256k1_field", "secp256k1_group", "secp256k1_prec", "secp256k1", 
                 "address", "mnemonic_constants", "int_to_address"];
    files.iter()
        .map(|f| fs::read_to_string(format!("./cl/{}.cl", f)).expect(&format!("Failed to read: {}", f)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     GPU BIP39 12-Word Permutation Scanner                  ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Target: 3CKkHm2nTS46vrTiGayj4fPtggjq8opcZF                  ║");
    println!("║ Words:  asset basket capital execute gauge improve         ║");
    println!("║         pair price require sell share trend                ║");
    println!("║ Total:  479,001,600 permutations                           ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let platform_id = core::default_platform().expect("No OpenCL platform found");
    let device_ids = core::get_device_ids(&platform_id, Some(ocl::flags::DEVICE_TYPE_GPU), None)
        .expect("No GPU found");
    
    println!("✅ Found {} GPU(s)", device_ids.len());
    
    let device_id = device_ids[0];
    let device_name = core::get_device_info(&device_id, core::DeviceInfo::Name)
        .map(|i| format!("{:?}", i))
        .unwrap_or_else(|_| "Unknown".to_string());
    println!("✅ Using: {}\n", device_name);
    
    let context_properties = ContextProperties::new().platform(platform_id);
    let context = core::create_context(Some(&context_properties), &[device_id], None, None)
        .expect("Failed to create context");
    
    println!("🔧 Compiling OpenCL kernels...");
    let src = CString::new(load_kernel_source()).expect("Failed to create kernel source");
    let program = core::create_program_with_source(&context, &[src])
        .expect("Failed to create program");
    
    if let Err(e) = core::build_program(&program, Some(&[device_id]), &CString::new("").unwrap(), None, None) {
        eprintln!("Kernel build error: {:?}", e);
        return;
    }
    println!("✅ Kernels compiled\n");
    
    let queue = core::create_command_queue(&context, &device_id, None)
        .expect("Failed to create command queue");
    
    let start_time = Instant::now();
    let mut checked: u64 = 0;
    let report_interval: u64 = 100_000;
    
    println!("🚀 Starting GPU search...\n");
    
    for k in 0..TOTAL_PERMS {
        let perm = permutation_to_indices(k);
        let (mnemonic_hi, mnemonic_lo) = encode_mnemonic(&perm);
        
        let mut target_mnemonic = vec![0u8; 120];
        let mut mnemonic_found = vec![0u8; 1];
        
        let target_buf = unsafe { 
            core::create_buffer(&context, flags::MEM_WRITE_ONLY | flags::MEM_COPY_HOST_PTR, 
                120, Some(&target_mnemonic)).unwrap()
        };
        let found_buf = unsafe {
            core::create_buffer(&context, flags::MEM_WRITE_ONLY | flags::MEM_COPY_HOST_PTR, 
                1, Some(&mnemonic_found)).unwrap()
        };
        
        let kernel = core::create_kernel(&program, "int_to_address").unwrap();
        core::set_kernel_arg(&kernel, 0, ArgVal::scalar(&mnemonic_hi)).unwrap();
        core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&mnemonic_lo)).unwrap();
        core::set_kernel_arg(&kernel, 2, ArgVal::mem(&target_buf)).unwrap();
        core::set_kernel_arg(&kernel, 3, ArgVal::mem(&found_buf)).unwrap();
        
        unsafe {
            core::enqueue_kernel(&queue, &kernel, 1, None, &[1, 1, 1], 
                None, None::<core::Event>, None::<&mut core::Event>).unwrap();
            core::enqueue_read_buffer(&queue, &found_buf, true, 0, &mut mnemonic_found, 
                None::<core::Event>, None::<&mut core::Event>).unwrap();
        }
        
        checked += 1;
        
        if mnemonic_found[0] == 0x01 {
            unsafe {
                core::enqueue_read_buffer(&queue, &target_buf, true, 0, &mut target_mnemonic, 
                    None::<core::Event>, None::<&mut core::Event>).unwrap();
            }
            let mnemonic = String::from_utf8_lossy(&target_mnemonic)
                .trim_matches(char::from(0))
                .to_string();
            
            println!("\n╔════════════════════════════════════════════════════════════╗");
            println!("║                    🎉 FOUND! 🎉                             ║");
            println!("╠════════════════════════════════════════════════════════════╣");
            println!("║ Mnemonic: {}", mnemonic);
            println!("║ Permutation #{}", k);
            println!("║ Time: {:.2}s", start_time.elapsed().as_secs_f64());
            println!("╚════════════════════════════════════════════════════════════╝");
            
            // Save to file
            let mut file = fs::File::create("/content/FOUND.txt").unwrap();
            writeln!(file, "MNEMONIC FOUND!").unwrap();
            writeln!(file, "Phrase: {}", mnemonic).unwrap();
            writeln!(file, "Permutation: {}", k).unwrap();
            writeln!(file, "Word order: {}", perm_to_words(&perm)).unwrap();
            
            return;
        }
        
        // Progress report
        if checked % report_interval == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = checked as f64 / elapsed;
            let remaining = TOTAL_PERMS - checked;
            let eta_secs = remaining as f64 / rate;
            let pct = (checked as f64 / TOTAL_PERMS as f64) * 100.0;
            
            print!("\r[{:.2}%] {:>10}/{} | {:.0}/s | ETA: {:.1}m    ", 
                pct, checked, TOTAL_PERMS, rate, eta_secs / 60.0);
            std::io::stdout().flush().unwrap();
        }
    }
    
    println!("\n\n❌ Search complete. Target not found.");
    println!("Checked {} permutations in {:.2}s", checked, start_time.elapsed().as_secs_f64());
}
