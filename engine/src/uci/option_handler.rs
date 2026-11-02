// Engine options the GUI can change with "setoption"
// each one gets advertised in the "uci" reply so the GUI knows it exists

use crate::search::searcher::DEFAULT_HASH_MB;

pub struct EngineOptions {
    pub hash_mb: usize,
    pub move_overhead_ms: i64, // time we keep back for lag / GUI delay, flagging on lichess because of ping is embarrassing
}

// what the engine has to actually do after an option changed
pub enum OptionChange {
    ResizeHash(usize),
    ClearHash,
    Nothing,
}

impl EngineOptions {

    pub fn new() -> Self {
        Self {
            hash_mb: DEFAULT_HASH_MB,
            move_overhead_ms: 30,
        }
    }

    pub fn print(&self) {
        println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 4096");
        println!("option name Clear Hash type button");
        println!("option name Threads type spin default 1 min 1 max 1"); // TODO: lazy SMP
        println!("option name Move Overhead type spin default 30 min 0 max 5000");
    }

    pub fn set(&mut self, name: &str, value: Option<&str>) -> OptionChange {
        // option names aren't case sensitive in the spec
        match name.to_lowercase().as_str() {
            "hash" => {
                if let Some(mb) = value.and_then(|v| v.parse::<usize>().ok()) {
                    self.hash_mb = mb.clamp(1, 4096);
                    return OptionChange::ResizeHash(self.hash_mb);
                }
            }
            "clear hash" => return OptionChange::ClearHash,
            "move overhead" => {
                if let Some(ms) = value.and_then(|v| v.parse::<i64>().ok()) {
                    self.move_overhead_ms = ms.clamp(0, 5000);
                }
            }
            "threads" => {} // only 1 for now, accept it so GUIs don't complain
            _ => println!("info string unknown option {name}"),
        }
        OptionChange::Nothing
    }
}

// Default state / restarted state
impl Default for EngineOptions {
    fn default() -> Self {
        Self::new()
    }
}
