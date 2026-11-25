use std::io::{self, Read, Write};
use std::process::ExitCode;

const MAX_PAYLOAD: usize = 4096;

fn main() -> ExitCode {
    let mut payload = Vec::new();
    if io::stdin()
        .take((MAX_PAYLOAD + 1) as u64)
        .read_to_end(&mut payload)
        .is_err()
        || payload.len() > MAX_PAYLOAD
        || payload.iter().any(|byte| !(b' '..=b'~').contains(byte))
    {
        return ExitCode::FAILURE;
    }

    let mut hash = 2_166_136_261u32;
    for byte in payload {
        hash = (hash ^ u32::from(byte)).wrapping_mul(16_777_619);
    }
    let mut stdout = io::stdout().lock();
    if writeln!(stdout, "{hash:08x}")
        .and_then(|()| stdout.flush())
        .is_err()
    {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
