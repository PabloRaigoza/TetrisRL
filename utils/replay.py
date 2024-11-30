import numpy as np
import cv2
import sys
from environment import makeBC 


# Checking command line arguments
if len(sys.argv) != 2:
    print("Usage: python replay.py <path_to_data>")
    sys.exit(1)


# Getting command line arguments
path = sys.argv[1]

# Extract the seed from the file path
seed = int(path.split('_')[3])

# save the data
data = np.load(path, allow_pickle=True)
# Sample data path 'data/BC/BC_data_000_551896_1000.npy' 

# Initialize the environment
env = makeBC()  # Use your custom Tetris environment
observation = env.reset(seed=seed)

# Replay the collected sequence
for step, record in enumerate(data):
    state = record['state']
    action = record['action']
    reward = record['reward']
    terminated = record['terminated']
    truncated = record['truncated']
    info = record['info']

    # Extract board and active tetromino mask
    if step == 0:
        active_tetromino_mask = state[0]['active_tetromino_mask']
        board = state[0]['board']
    else:
        active_tetromino_mask = state['active_tetromino_mask']
        board = state['board']
    
    # Create a visualization of the board
    board_height, board_width = board.shape
    cell_size = 30  # Size of each cell in pixels for visualization
    canvas = np.zeros((board_height * cell_size, board_width * cell_size, 3), dtype=np.uint8)

    # Map board values to colors (adjust as per your board encoding)
    for r in range(board_height):
        for c in range(board_width):
            cell_value = board[r, c]
            if cell_value == 0:  # Empty cell
                color = (50, 50, 50)  # Dark gray
            else:  # Occupied cell
                color = (0, 255, 0)  # Green
            
            # Draw the cell on the canvas
            top_left = (c * cell_size, r * cell_size)
            bottom_right = ((c + 1) * cell_size, (r + 1) * cell_size)
            cv2.rectangle(canvas, top_left, bottom_right, color, -1)
    
    # Overlay the active tetromino
    for r in range(board_height):
        for c in range(board_width):
            if active_tetromino_mask[r, c] == 1:
                top_left = (c * cell_size, r * cell_size)
                bottom_right = ((c + 1) * cell_size, (r + 1) * cell_size)
                cv2.rectangle(canvas, top_left, bottom_right, (255, 0, 0), -1)  # Blue for active tetromino

    # Display the board
    cv2.imshow("Tetris Replay", canvas)

    # Pause to mimic real-time replay
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

    # Step through the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # Check if the game is over
    if terminated or truncated or step == len(data) - 1:
        print(f"Game Over at step {step+1}.")
        break

# Close environment and release resources
env.close()
cv2.destroyAllWindows()