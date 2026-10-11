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
    state: State,
    action: Action,
    _next_state: State,
    reward: f32,
}


// Q-Value
/*
 *
 */
fn q_value<'a>(s: &State, a: &Action, q_table: &'a mut HashMap<(State, Action), f32>) -> &'a mut f32 {
    let q = q_table.get_mut(&(s.clone(), a.clone())).ok_or(0).unwrap();
    q
}

// Alpha (α)
/*
 * Learning rate
 */
const ALPHA: f32 = 0.1;



/*
 * TD Update
 * Q-values are updated from individual transitions.
 *
 * TD target
 *   target = R + γQ(s', a')
 *
 * TD error
 *   δ = target - Q(s, a)
 *
 * Q-value update
 *   Q(s, a) <- Q(s, a) + α * δ
 */
fn td_update (
    step: Step,
    next_value: f32,
    q_table: &mut HashMap<(State, Action), f32>,
) {
    // Q(s, a)
    let q = q_value(&step.state, &step.action, q_table);

    // target = R + γQ(s', a')
    let target = step.reward + GAMMA * next_value;

    // δ = target - Q(s, a)
    let td_error = target - *q;

    // Q(s, a) <- Q(s, a) + α * δ
    *q += ALPHA * td_error;
}

fn action(s: &State, actions: &Vec<Action>, q_table: &mut HashMap<(State, Action), f32>) -> Action {
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
                let mut s: State = initial_s.clone(); // s

                while !(s == *terminal_s) {
                    let a = action(&s, &actions, &mut q_table); // a

                    // E(s, a) -> (s', r)
                    let (s_next, r) = Environment::step(&s, &a);

                    let step = Step {
                        state: s,
                        action: a,
                        _next_state: s_next.clone(),
                        reward: r,
                    };

                    // This value must be taken from Q table
                    // Out of all possible actions one of the (s, a) pair must be selected
                    // That part is handled in specific TD algorithm like SARSA or Q-learning
                    let next_value = 0.0;

                    // TD update
                    td_update(step, next_value, &mut q_table);

                    s = s_next;
                }
            }
        }
    }
}
