# Space Invaders
Playing Atari using Deep Reinforcement Learning

![game](https://www.gymlibrary.ml/_images/space_invaders.gif)

# Algorithm
Deep Q-Learning (DQN) with Experience Replay and an $\epsilon$-greedy strategy (https://arxiv.org/pdf/1312.5602.pdf).

Pseudocode:
1. Initialize replay memory $D$ with capacity $N$
2. Initialize Q-function with random weights
3. For $episode = 1...M$, do:
    * Intialize sequence $ s_1 = \{x_1\} $
    * For $ t = 1... T $, do:
        * With probability $\epsilon$ select a random action $a_t$, otherwise select $ a_t = argmax_a Q^*(s_t, a; \theta) $
        * Pass the action to the emulator and observe reward $r_t$ and state $x_{t+1}$
        * Store the transition $ (s_t, a_t, r_t, s_{t+1}) $ in $ D $
        * Sample random minibatch of transitions $ (s_j , a_j , r_j , s_{j+1}) $ from $ D $
        * Assign $$ y_j =
            \begin{cases}
            r_j & \text{for a terminal } s_{j+1} \\
            r_j + \gamma \max_{a'} Q^*(s_{j+1}, a'; \theta) & \text{for a non-terminal } s_{j+1}
            \end{cases} $$
        * Perform a gradient descent step on $ (y_j − Q(s_j, a_j; \theta))^2 $:
$$ \nabla_{\theta_i}L_i(\theta_i) = E_{s,a \sim p(·); s' \sim \epsilon} [(r + \gamma\max_a'Q(s', a'; \theta_{i-1}) - Q(s, a, \theta_i))\nabla_{\theta_i}Q(s, a; \theta_i)] $$
$$ L_i(\theta_i) = E_{s,a \sim p(·)}[(y_i - Q(s, a; \theta_i))^2] $$
$$ Q^*(s, a) = E_{s' \sim \epsilon}[r + \gamma\max_a'Q^*(s', a') \mid s, a] $$

### PDF Generation
```bash
pip install "nbconvert[webpdf]"
jupyter nbconvert --to webpdf --allow-chromium-download atari.ipynb
```
