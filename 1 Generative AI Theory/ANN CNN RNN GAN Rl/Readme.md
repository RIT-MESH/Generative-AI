## Artificial Neural Network (ANN)
### Structure and Mechanism
An Artificial Neural Network (ANN) consists of:
- **Input Layer**: Receives raw data inputs.
- **Hidden Layers**: Processes the data using weighted connections and activation functions.
- **Output Layer**: Produces the final result.

### Step-by-Step Working
1. **Data Preprocessing**: Normalize and prepare the input data.
2. **Forward Propagation**: Input data passes through the network, applying weighted sums and activation functions.
3. **Loss Calculation**: Compare the network’s output with the true output using a loss function.
4. **Backpropagation**: Compute gradients and update weights using optimization techniques like SGD or Adam.
5. **Iteration & Training**: Repeat steps 2-4 over multiple epochs until optimal performance is achieved.
<p align="center"><img src="https://github.com/RIT-MESH/Generative-AI/blob/main/1%20Generative%20AI%20Theory/ANN%20CNN%20RNN%20GAN%20Rl/ann.png?raw=true"alt="Sublime's custom image" width="900"/>
 </p>
 
### Applications
- **Healthcare**: Disease detection from medical images, patient risk prediction.
- **Finance**: Credit scoring, fraud detection, and stock price prediction.
- **Marketing**: Customer behavior analysis, targeted advertising, and recommendation systems.
- **Energy and Manufacturing**: Predictive maintenance, quality control, and load forecasting in power grids.

---

## Convolutional Neural Network (CNN)
### How it Works
CNNs specialize in processing structured grid-like data, mainly images.

### Step-by-Step Working
1. **Input Image**: The image is converted into numerical pixel values.
2. **Convolution Operation**: Filters (kernels) scan over the image to extract features.
3. **Activation Function (ReLU)**: Applies non-linearity to the feature maps.
4. **Pooling (Max/Average)**: Reduces dimensionality while retaining important features.
5. **Flattening**: Converts the pooled feature map into a single vector.
6. **Fully Connected Layers**: Processes the features for final classification.
7. **Softmax/Output Layer**: Produces probabilities for classification labels.
<p align="center"><img src="https://github.com/RIT-MESH/Generative-AI/blob/main/1%20Generative%20AI%20Theory/ANN%20CNN%20RNN%20GAN%20Rl/cnn.png?raw=true"alt="Sublime's custom image" width="900"/>
 </p>

### Applications
- **Medical Imaging**: Tumor detection in MRI, CT scans, and X-rays.
- **Self-Driving Cars**: Object detection, lane detection, and scene segmentation.
- **Security**: Facial recognition, surveillance, and anomaly detection.
- **Speech Processing**: Speech-to-text conversion using spectrograms.
- **Augmented Reality (AR) & Virtual Reality (VR)**: Gesture recognition and scene reconstruction.

---

## Recurrent Neural Network (RNN)
### Recurrent Structure
RNNs are designed for sequential data by maintaining an internal state (memory) that captures dependencies across time steps.

### Step-by-Step Working
1. **Input Sequence**: A sequence of data (text, speech, time-series) is fed into the RNN.
2. **Hidden State Update**: The network maintains memory of previous inputs.
3. **Recurrent Processing**: Each input step influences future computations.
4. **Final Prediction**: The last hidden state is used to generate an output.
5. **Backpropagation Through Time (BPTT)**: Gradients are computed and weights are updated to improve predictions.
<p align="center"><img src="https://github.com/RIT-MESH/Generative-AI/blob/main/1%20Generative%20AI%20Theory/ANN%20CNN%20RNN%20GAN%20Rl/rnn.png?raw=true"alt="Sublime's custom image" width="900"/>
 </p>
   

### Applications
- **Natural Language Processing (NLP)**: Sentiment analysis, text generation, machine translation, chatbots.
- **Speech Recognition**: Voice-to-text conversion in digital assistants like Siri, Google Assistant.
- **Finance**: Stock market prediction, economic trend analysis, anomaly detection in transaction data.
- **Healthcare**: ECG signal processing, disease progression modeling, and medical report generation.
- **Music & Video Analysis**: Music composition, video captioning, and gesture recognition.

---

## Generative Adversarial Network (GAN)
### Generator-Discriminator Architecture
GANs consist of two neural networks:
- **Generator**: Creates fake data resembling real samples.
- **Discriminator**: Distinguishes real data from generated data.
- Training involves adversarial learning where the generator improves over time by deceiving the discriminator, and the discriminator continuously improves its classification ability.

### Step-by-Step Working
1. **Random Noise Input**: Generator starts with a random noise vector.
2. **Synthetic Data Generation**: The generator processes the noise to create fake data.
3. **Discriminator Evaluation**: Discriminator classifies the data as real or fake.
4. **Loss Computation**: The difference between predictions and actual labels is measured.
5. **Backpropagation**: Generator improves by reducing the ability of the discriminator to distinguish real from fake data.
6. **Repeat Until Convergence**: Both networks continue learning until the generated data is indistinguishable from real data.
<p align="center"><img src="https://github.com/RIT-MESH/Generative-AI/blob/main/1%20Generative%20AI%20Theory/ANN%20CNN%20RNN%20GAN%20Rl/gan.png?raw=true"alt="Sublime's custom image" width="900"/>
 </p>


### Applications
- **Image Generation**: Creating realistic human faces (e.g., StyleGAN), AI-generated art.
- **Image-to-Image Translation**: Colorizing black-and-white images, converting sketches to realistic photos.
- **Healthcare**: Generating synthetic patient data for research, improving image quality in MRI and CT scans.
- **Gaming**: Procedural content generation, character design, environment creation.
- **Fashion & E-Commerce**: Virtual clothing try-on, AI-generated fashion designs.
- **Deepfake Technology**: Modifying faces in videos for entertainment and ethical AI applications.

---

## Reinforcement Learning (RL)
### Basic Principles
- **Agent**: Learns to make decisions by interacting with an environment.
- **Environment**: Provides feedback based on actions taken by the agent.
- **Reward Function**: Guides the learning process to optimize long-term outcomes.
- **Exploration vs. Exploitation**: Balances discovering new strategies (exploration) vs. using known optimal strategies (exploitation).
- **Markov Decision Process (MDP)**: Defines RL environments using states, actions, rewards, and transitions.

### Step-by-Step Working
1. **Initialize Agent**: The agent starts with no prior knowledge of the environment.
2. **Observe State**: The agent perceives the current state of the environment.
3. **Select Action**: Based on a policy, the agent chooses an action.
4. **Receive Reward**: The environment provides feedback (positive/negative reward).
5. **Update Policy**: The agent updates its decision-making strategy based on rewards.
6. **Repeat**: The process continues until the agent learns an optimal policy.
<p align="center"><img src="https://github.com/RIT-MESH/Generative-AI/blob/main/1%20Generative%20AI%20Theory/ANN%20CNN%20RNN%20GAN%20Rl/rl.png?raw=true"alt="Sublime's custom image" width="900"/>
 </p>


### Applications
- **Gaming**: AI systems mastering complex games (e.g., AlphaGo, OpenAI Five, Dota 2, StarCraft II bots).
- **Robotics**: Training robotic arms for precision tasks, autonomous drone navigation.
- **Healthcare**: Optimizing treatment plans for personalized medicine, robotic surgery.
- **Finance**: Reinforcement-based trading strategies, portfolio management, fraud detection.
- **Autonomous Vehicles**: RL-based driving policies, adaptive traffic signal control.
- **Smart Cities & IoT**: Energy-efficient HVAC systems, smart grid optimization, and supply chain logistics.

Each of these deep learning paradigms plays a crucial role in advancing AI across industries, often complementing one another in complex real-world applications.

