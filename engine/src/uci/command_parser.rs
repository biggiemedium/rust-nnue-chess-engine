// Turns the text a GUI sends us into a Command
// http://wbec-ridderkerk.nl/html/UCIProtocol.html
//
// Everything is space separated so it's just walking tokens, unknown stuff gets ignored
// (the spec says to ignore anything we don't understand instead of crashing)

#[derive(Debug, PartialEq)]
pub enum Command {
    Uci,
    IsReady,
    UciNewGame,
    SetOption { name: String, value: Option<String> },
    Position { fen: Option<String>, moves: Vec<String> }, // fen None = startpos
    Go(GoParams),
    Stop,
    Quit,

    // not uci, just for us
    Bench(Option<i32>), // "bench 8" -> depth
    Perft(u32),         // "go perft 5"
    Display,            // "d" -> print the fen
    Eval,               // "eval" -> static eval of the current position

    Unknown(String),
}

#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct GoParams {
    // times in ms, i64 because some GUIs send negative times when you're flagging
    pub wtime: Option<i64>,
    pub btime: Option<i64>,
    pub winc: Option<i64>,
    pub binc: Option<i64>,
    pub movestogo: Option<i64>,
    pub movetime: Option<i64>,
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub infinite: bool,
}

pub fn parse(line: &str) -> Command {
    let tokens: Vec<&str> = line.split_whitespace().collect();

    match tokens.first().copied() {
        Some("uci") => Command::Uci,
        Some("isready") => Command::IsReady,
        Some("ucinewgame") => Command::UciNewGame,
        Some("setoption") => parse_setoption(&tokens[1..]),
        Some("position") => parse_position(&tokens[1..]),
        Some("go") => parse_go(&tokens[1..]),
        Some("stop") => Command::Stop,
        Some("quit") => Command::Quit,
        Some("bench") => Command::Bench(tokens.get(1).and_then(|d| d.parse().ok())),
        Some("d") => Command::Display,
        Some("eval") => Command::Eval,
        _ => Command::Unknown(line.trim().to_string()),
    }
}

// setoption name <id> [value <x>]
// names can have spaces in them ("Clear Hash", "Move Overhead") -> everything between name and value
fn parse_setoption(tokens: &[&str]) -> Command {
    let value_at = tokens.iter().position(|&t| t == "value");
    let name_end = value_at.unwrap_or(tokens.len());

    let name = tokens.get(1..name_end).unwrap_or(&[]).join(" "); // skip the "name" token
    let value = value_at.map(|i| tokens[i + 1..].join(" "));

    Command::SetOption { name, value }
}

// position [startpos | fen <6 fields>] [moves <m1> <m2> ...]
fn parse_position(tokens: &[&str]) -> Command {
    let moves_at = tokens.iter().position(|&t| t == "moves");
    let setup_end = moves_at.unwrap_or(tokens.len());

    let fen = match tokens.first() {
        Some(&"fen") => Some(tokens[1..setup_end].join(" ")),
        _ => None, // "startpos" (or garbage -> startpos is the safest guess)
    };

    let moves = match moves_at {
        Some(i) => tokens[i + 1..].iter().map(|m| m.to_string()).collect(),
        None => Vec::new(),
    };

    Command::Position { fen, moves }
}

fn parse_go(tokens: &[&str]) -> Command {
    let mut go = GoParams::default();
    let mut i = 0;

    // every keyword is followed by its number except "infinite"
    while i < tokens.len() {
        let next = tokens.get(i + 1).copied().unwrap_or("");

        match tokens[i] {
            "perft" => return Command::Perft(next.parse().unwrap_or(1)),
            "infinite" => {
                go.infinite = true;
                i += 1;
                continue;
            }
            "wtime" => go.wtime = next.parse().ok(),
            "btime" => go.btime = next.parse().ok(),
            "winc" => go.winc = next.parse().ok(),
            "binc" => go.binc = next.parse().ok(),
            "movestogo" => go.movestogo = next.parse().ok(),
            "movetime" => go.movetime = next.parse().ok(),
            "depth" => go.depth = next.parse().ok(),
            "nodes" => go.nodes = next.parse().ok(),
            _ => {
                // ponder, searchmoves, mate ... not supported, skip the word
                i += 1;
                continue;
            }
        }
        i += 2;
    }

    Command::Go(go)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_startpos_with_moves() {
        assert_eq!(
            parse("position startpos moves e2e4 e7e5"),
            Command::Position { fen: None, moves: vec!["e2e4".into(), "e7e5".into()] }
        );
    }

    #[test]
    fn position_fen() {
        let fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1";
        assert_eq!(
            parse(&format!("position fen {fen} moves e1e2")),
            Command::Position { fen: Some(fen.into()), moves: vec!["e1e2".into()] }
        );
    }

    #[test]
    fn go_with_clock() {
        let Command::Go(go) = parse("go wtime 60000 btime 59000 winc 1000 binc 1000 movestogo 20") else {
            panic!("not a go");
        };
        assert_eq!((go.wtime, go.btime, go.winc, go.binc, go.movestogo), (Some(60000), Some(59000), Some(1000), Some(1000), Some(20)));
        assert!(!go.infinite);
    }

    #[test]
    fn go_infinite_and_depth() {
        let Command::Go(go) = parse("go infinite depth 12") else {
            panic!("not a go");
        };
        assert!(go.infinite);
        assert_eq!(go.depth, Some(12));
    }

    #[test]
    fn setoption_with_spaces() {
        assert_eq!(
            parse("setoption name Move Overhead value 50"),
            Command::SetOption { name: "Move Overhead".into(), value: Some("50".into()) }
        );
        assert_eq!(
            parse("setoption name Clear Hash"),
            Command::SetOption { name: "Clear Hash".into(), value: None }
        );
    }

    #[test]
    fn perft_and_unknown() {
        assert_eq!(parse("go perft 4"), Command::Perft(4));
        assert_eq!(parse("banana"), Command::Unknown("banana".into()));
    }
}
