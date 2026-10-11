use std::collections::HashMap;

use crate::{State, Action, GAMMA};


// Environemnt
/*
 * E(s, a) -> (s', r)
 */
struct Environment;

impl Environment {
    pub fn step(_s: &State, _a: &Action) -> Step {
        Step {
            next_state: State,
            reward: -1.0,
        }
    }
}

struct Step {
    next_state: State,
    reward: f32,
}


// Policy (π) - defined
/*
 *   π(a | s)
 */
fn policy(_s: &State, _a: &Action) -> f32 {
    0.9
}

// Value (V) - learnt
/*
 * Value Read
 *   V(s) -> v
 */
fn value(s: &State, v_table: &HashMap<State, f32>) -> f32 {
    let v = v_table[s];
    v
}

// Alpha (α)
/*
 *
 */
const ALPHA: f32 = 0.9;



/*
 * V(s') <- V(s') + α[G(s') - V(s')]
 */
fn monte_carlo_learning(
    episode: Vec<Step>,
    v_table: &mut HashMap<State, f32>,
) {
    // Calculate returns and update V backwards.
    let mut g = 0.0;

    for step in episode.iter().rev() {
        // G(s') = R(s') + γG(s')
        g = step.reward + GAMMA * g;

        // V(s')
        let old_value = value(&step.next_state, v_table);

        // V(s') <- V(s') + α[G(s') - V(s')]
        let new_value = old_value + ALPHA * (g - old_value);

        v_table.insert(step.next_state.clone(), new_value);
    }
}

fn action(s: &State, actions: &Vec<Action>) -> Action {
    //  An action must be selected from a probability distribution
    //    based on π(a | s) for each action
    for a in actions {
        _ = policy(s, a);
    }

    Action
}

pub fn run() {
    let states: Vec<State> = Vec::new();
    let actions: Vec<Action> = Vec::new();
    let mut v_table: HashMap<State, f32> = HashMap::new();

    // Train
    for initial_s in &states {
        for terminal_s in &states {
            for _ in 0..50 {
                // Generate one complete episode
                let mut episode: Vec<Step> = Vec::new();
                let mut s: State = initial_s.clone(); // s

                while !(s == *terminal_s) {
                    let a = action(&s, &actions); // a

                    // E(s, a) -> (s', r)
                    let Step {next_state: s_next, reward: r} = Environment::step(&s, &a);

                    // Record the observed state and reward.
                    episode.push(
                        Step {
                            next_state: s_next.clone(),
                            reward: r,
                        }
                    );

                    s = s_next;
                }

                //  Learn from the episode
                monte_carlo_learning(episode, &mut v_table)
            }
        }
    }
}
