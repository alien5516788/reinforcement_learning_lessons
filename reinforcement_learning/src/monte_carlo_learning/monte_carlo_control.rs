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

// Episode Step
struct Step {
    state: State,   // Current state
    action: Action, // Action took from current state
    reward: f32,    // Reward recieved from next state
}


// Q-Value
/*
 *
 */
fn q_value(s: &State, a: &Action, q_table: &HashMap<(State, Action), f32>) -> f32 {
    let q = q_table[&(s.clone(), a.clone())];
    q
}

// Alpha (α)
/*
 * Learning rate
 */
const ALPHA: f32 = 0.1;



/*
 * Q-Value is learnt by calculating discounted return backward from the end of the episode
 *   then updating the q-value of each step state using the return
 *
 *   G_t = R_t+1 + γG_t+1
 *   Q(s_t, a_t) <- Q(s_t, a_t) + α[G_t - Q(s_t, a_t)]
 */
fn monte_carlo_control (
    episode: Vec<Step>,
    q_table: &mut HashMap<(State, Action), f32>
) {
    // g
    let mut g = 0.0;

    // We start as t+1 = terminal step which is not included in the episode
    //   and t = last step in the episode
    for step in episode.iter().rev() {
        // G_t = R_t+1 + γG_t+1
        g = step.reward + GAMMA * g;

        // Q(s_t, a_t)
        let mut q = q_value(&step.state, &step.action, q_table);

        // q
        q = q + ALPHA * (g - q);

        // Q(s_t, a_t) <- Q(s_t, a_t) + α[G_t - Q(s_t, a_t)]
        q_table.insert((step.state.clone(), step.action.clone()), q);
    }
}

fn action(s: &State, actions: &Vec<Action>, q_table: &HashMap<(State, Action), f32>) -> Action {
    //  An action must be selected from a probability distribution
    //    based on Q(s, a) for each action
    for a in actions {
        _ = q_value(s, a, q_table);
    }

    Action
}

pub fn run() {
    let states: Vec<State> = Vec::new();
    let actions: Vec<Action> = Vec::new();
    let mut q_table: HashMap<(State, Action), f32> = HashMap::new();

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
                    let a = action(&s, &actions, &q_table); // a

                    // E(s, a) -> (s', r)
                    let (s_next, r) = Environment::step(&s, &a);

                    episode.push(
                        Step {
                            state: s,
                            action: a,
                            reward: r,
                        }
                    );

                    s = s_next;
                }

                //  Learn from the episode
                monte_carlo_control(episode, &mut q_table)
            }
        }
    }
}
