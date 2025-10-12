import numpy as np
import imageio


def evaluate(model, env, num_episodes=10, video_path=None):
    """
    Evaluate a trained RL model on the given environment.
    Optionally record and save a video of the agent's performance.
    """
    all_rewards = []
    frames = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=np.random.randint(0, 2**32 - 1))
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if video_path:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        all_rewards.append(ep_reward)
        print(f"Episode {ep + 1}: reward = {ep_reward}")

    avg_reward = np.mean(all_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")

    if video_path and len(frames) > 0:
        print(f"Saving video to {video_path}")
        imageio.mimsave(video_path, frames, fps=10)

    return avg_reward
