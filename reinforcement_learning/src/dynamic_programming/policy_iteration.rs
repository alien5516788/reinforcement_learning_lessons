use std::collections::HashMap;

use crate::{State, Action, world_model, reward, GAMMA};


// Policy (π) - learnt
/*
 *   π(a | s)
 */
fn policy(s: &State, a: &Action, pi_table: &mut HashMap<(State, Action), f32>) -> f32 {
    let sa = (s.clone(), a.clone());
    let p = pi_table[&sa];
    p
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
    pi_table: &mut HashMap<(State, Action), f32>,
    v_table: &mut HashMap<State, f32>
) {
    let mut new_table: HashMap<State, f32> = HashMap::new();

    for s in states {
        // v
        let mut v = 0.0;

        // Σ_a
        for a in actions {
            // π(a | s)
            let action_probability = policy(s, a, pi_table);

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



/*
 * Policy is improved by calculating the Q(s,a) for every possible action a and
 *   updating the policy table with the action a* (action a with best Q value)
 *
 *     Q(s, a) = Σ_s' P(s' | s, a) * [R(s, a, s') + γV(s')]
 *     a* = argmax_a Q(s, a)
 *     π(a | s) <- 1 if a == a*; 0 otherwise                    // greedy/deterministic
 *     π(a | s) <- 0.9 if a == a*; 0.1 * (1 / n - 1) otherwise  // stochastic
 */
fn policy_iteration(
    states: &Vec<State>,
    actions: &Vec<Action>,
    pi_table: &mut HashMap<(State, Action), f32>,
    v_table: &mut HashMap<State, f32>
) {
    // Policy evaluation
    for _ in 0..50 {
        policy_evaluation(&states, &actions, pi_table, v_table);
    }

    // Policy improvement

    let mut new_table: HashMap<(State, Action), f32> = HashMap::new();

    for s in states {
        let mut best_action = actions[0].clone(); // a*
        let mut best_action_value = f32::NEG_INFINITY;   // max_a Q(s, a)

        for a in actions {
            // q
            let mut q = 0.0;

            // Σ_s'
            for s_next in states {

                // P(s' | s,a)
                let transition_probability = world_model(s, a, s_next);

                // R(s, a, s')
                let immediate_reward = reward(s, a, s_next);

                // V(s')
                let future_value = value(s_next, v_table);

                // q += P(s' | s, a) * [R(s, a, s') + γV(s')]
                q += transition_probability * (immediate_reward + (GAMMA * future_value));
            }

            // Q(s, a) = q

            // a* <- arg_max_a Q(s, a)
            if q > best_action_value {
                best_action = a.clone();   // a*
                best_action_value = q; // max_a Q(s, a)
            }
        }

        // π(a | s) <- 1 if a == a*, 0 otherwise
        for a in actions {
            if *a == best_action {
                new_table.insert((s.clone(), a.clone()), 1.0);
            } else {
                new_table.insert((s.clone(), a.clone()), 0.0);
            };
        }
    }

    *pi_table = new_table;
}




pub fn run() {
    let states: Vec<State> = Vec::new();
    let actions: Vec<Action> = Vec::new();
    let mut pi_table: HashMap<(State, Action), f32> = HashMap::new();
    let mut v_table: HashMap<State, f32> = HashMap::new();

    // Train
    for _ in 0..50 {
        policy_iteration(&states, &actions, &mut pi_table, &mut v_table);
    }
}
