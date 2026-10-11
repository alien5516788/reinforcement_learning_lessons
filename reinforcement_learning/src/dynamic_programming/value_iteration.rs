use std::collections::HashMap;

use crate::{State, Action, world_model, reward, GAMMA};


// Value (V) - learnt
/*
 * Value Read
 *   V(s) -> v
 */
fn value(s: &State, v_table: &HashMap<State, f32>) -> f32 {
    let v = v_table[s];
    v
}

// Value iteration
/*
 * Value is learnt by continuously improving the value of each state
 * by choosing the action with the highest expected return.
 *
 *     Q(s, a) = Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
 *     V(s) <- max_a Q(s, a)
 */
fn value_iteration(
    states: &Vec<State>,
    actions: &Vec<Action>,
    v_table: &mut HashMap<State, f32>
) {
    let mut new_table: HashMap<State, f32> = HashMap::new();

    for s in states {
        // v
        let mut v = f32::NEG_INFINITY; // max_a Q(s, a)

        for a in actions {
            // v
            let mut action_value = 0.0;

            // Σ_s'
            for s_next in states {

                // P(s' | s, a)
                let transition_probability = world_model(s, a, s_next);

                // R(s, a, s')
                let immediate_reward = reward(s, a, s_next);

                // V(s')
                let future_value = value(s_next, v_table);

                // Add the next state contribution to Q(s, a)
                action_value += transition_probability * (immediate_reward + (GAMMA * future_value));
            }

            // Q(s, a) = q

            // V(s) <- max_a Q(s, a)
            if action_value > v {
                v = action_value; // max_a Q(s, a)
            }
        }

        // V(s) = v
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
        value_iteration(&states, &actions, &mut v_table);
    }
}
