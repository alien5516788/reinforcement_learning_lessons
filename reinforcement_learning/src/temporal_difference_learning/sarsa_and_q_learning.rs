// The general TD algorithm doesn't define how the next value is calculated
// It depends on the specific TD implementation

/*
 * SARSA (S -> A -> R -> S -> A)
 *   - SARSA calculate the next value based on the action selected using policy
 *
 *     a' = action(s', A)
 *     next_value = Q(s', a')
 *
 * Q-Learning
 *   - Q-Learning calculates the next value using highest Q-value
 *     among all available actions for the next state
 *
 *       next_value = max_a Q(s', a')
 */
