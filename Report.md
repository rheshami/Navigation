
    # Learning algorithm
    For this project I used basic DQN algorithm with following parameters

    - Maximum number of timesteps per episode: 1000
    - Starting value of epsilon, for epsilon-greedy action selection: 0.1
    - Minimum value of epsilon: 0.001
    - Multiplicative factor (per episode) for decreasing epsilon (epsilon decay): 0.995

    I used following neural network model:

    - Fully connected layer - input size: 37 (state size) output size: 124
    - Fully connected layer - input: 124 output 124
    - Fully connected layer - input: 124 output 64
    - Fully connected layer - input: 64 output: 4 (action size)
    - I also used dropout to avoid over fitting.

    After less than 450 episodes agent reached average score of +13.0
    ```
    Episode 100\tAverage Score: 0.73
    Episode 200\tAverage Score: 2.56
    Episode 300\tAverage Score: 7.37
    Episode 400\tAverage Score: 9.63
    Episode 500\tAverage Score: 12.41
    Episode 543\tAverage Score: 13.02
    Environment solved in 443 episodes!\tAverage Score: 13.02
    ```
    ![result](resources/plot1.png)

    ### Untrained Agent
    ![untrained](resources/untrained.gif)

    ### Trained Agent
    ![trained](resources/trained.gif)

    # Future Work Ideas
    - Train the agent to learn directly from pixels
    - Create a new Environment with different score and states and use same learning model.
    - Experiment with other algorithms such as Double DQN or Dueling DQN then compare the results.