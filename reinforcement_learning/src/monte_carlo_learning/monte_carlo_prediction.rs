use std::collections::HashMap;

use crate::{State, Action, GAMMA};


// Environemnt
/*
 * E(s, a) -> (s', r)
 */
struct Environment;

impl Environment {
    pub fn step(_s: &State, _a: &Action) -> (State, f32) {
        (State, -1.0)
    }
}

// Episodic Step
struct Step {
    state: State, // Current state
    reward: f32,  // Reward for leaving current state
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
const ALPHA: f32 = 0.9;



/*
 * Value is learnt by calculating discounted return backward from the end of the episode
 *   then updating the value of each step state using the return
 *
 *   G_t = R_t+1 + γG_t+1
 *   V(s_t) <- V(s_t) + α[G_t - V(s_t)]
 */
fn monte_carlo_prediction (
    episode: Vec<Step>,
    v_table: &mut HashMap<State, f32>,
) {
    // g
    let mut g = 0.0;

    // We start as t+1 = terminal step which is not included in the episode
    //   and t = last step in the episode
    for step in episode.iter().rev() {
        // G_t = R_t+1 + γG_t+1
        g = step.reward + GAMMA * g;

        // V(s_t)
        let mut v = value(&step.state, v_table);

        // v
        v = v + ALPHA * (g - v);

        // V(s_t) <- V(s_t) + α[G_t - V(s_t)]
        v_table.insert(step.state.clone(), v);
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
            // In reality initial state and terminal state depends on the environment
            // All possible initial and terminal states may not be explored

            for _ in 0..50 {
                // Generate one complete episode
                let mut episode: Vec<Step> = Vec::new();
                let mut s: State = initial_s.clone(); // s

                while !(s == *terminal_s) {
                    let a = action(&s, &actions); // a

                    // E(s, a) -> (s', r)
                    let (s_next, r) = Environment::step(&s, &a);

                    episode.push(
                        Step {
                            state: s,
                            reward: r,
                        }
                    );

                    s = s_next;
                }

                //  Learn from the episode
                monte_carlo_prediction(episode, &mut v_table)
            }
        }
    }
}
