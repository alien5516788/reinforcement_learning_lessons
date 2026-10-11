use std::collections::HashMap;

use crate::{State, Action, world_model, reward, GAMMA};


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

/*
 * Value Backup / Bellman equation for Policy Evaluation
 *     V(s) <- Σ_a π(a | s) * Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
 *
 * Value is learnt by continuously interacting with the world_model
 */
fn policy_evaluation(
    states: &Vec<State>,
    actions: &Vec<Action>,
    v_table: &mut HashMap<State, f32>
) {
    let mut new_table: HashMap<State, f32> = HashMap::new();

    for s in states {
        // v
        let mut v = 0.0;

        // Σ_a
        for a in actions {
            // π(a | s)
            let action_probability = policy(s, a);

            // Σ_s'
            for s_next in states {
                // P(s' | s, a)
                let transition_probability = world_model(s, a, s_next);

                // R(s, a, s')
                let immediate_reward = reward(s, a, s_next);

                // V(s')
                let future_value = value(&s_next, v_table);

                // v += π(a | s) * Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
                v += action_probability * transition_probability * (immediate_reward + (GAMMA * future_value));
            }
        }

        // V(s) <- v
        new_table.insert(s.clone(), v);
    }

    *v_table = new_table;
}


pub fn run() {
    let states: Vec<State> = Vec::new();
    let actions: Vec<Action> = Vec::new();
    let mut v_table: HashMap<State, f32> = HashMap::new();

    // Train
    for _ in 0..50 {
        policy_evaluation(&states, &actions, &mut v_table);
    }
}
