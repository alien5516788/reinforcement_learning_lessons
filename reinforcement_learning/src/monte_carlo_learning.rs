use std::collections::HashMap;

// One observed step in an episode.
struct Step {
    state: State,
    reward: f32,
}

// Monte Carlo prediction.
fn monte_carlo_learning(
    states: &[State],
    actions: &[Action],
    episodes: usize,
    alpha: f32,
    gamma: f32,
) {
    // Initialize the value function.
    let mut v_table: HashMap<State, f32> = HashMap::new();

    for state in states {
        v_table.insert(*state, 0.0);
    }

    // Generate multiple episodes.
    for _ in 0..episodes {
        let mut episode: Vec<Step> = Vec::new();

        let mut state = initial_state();

        // Generate one complete episode.
        while !is_terminal(&state) {
            // Sample an action using π(a | s).
            let chosen_action = action(&state, actions);

            // Let the environment produce the outcome.
            let (next_state, reward) =
                environment_step(&state, &chosen_action);

            // Record the observed state and reward.
            episode.push(Step {
                state,
                reward,
            });

            state = next_state;
        }

        // Calculate returns and update V backwards.
        let mut g = 0.0;

        for step in episode.iter().rev() {
            // G_t = R_(t+1) + γG_(t+1)
            g = step.reward + gamma * g;

            // V(S_t) ← V(S_t) + α[G_t - V(S_t)]
            let old_value = v_table[&step.state];

            let new_value =
                old_value + alpha * (g - old_value);

            v_table.insert(step.state, new_value);
        }
    }

    print_v_table(&v_table);
}
