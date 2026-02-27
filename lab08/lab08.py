import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# clean 255
# goal 200

def generate_enclosed_map(rows=15, cols=15, wall_density=0.2, trap_density=0.):
    # Start with a clean white path (255)
    grid = np.full((rows, cols), 255, dtype=np.uint8)
    
    grid[0, :] = 0   # Top wall
    grid[-1, :] = 0  # Bottom wall
    grid[:, 0] = 0   # Left wall
    grid[:, -1] = 0  # Right wall
    
    interior_shape = (rows-2, cols-2)
    random_roll = np.random.rand(*interior_shape)
    
    interior_mask = np.full((rows, cols), False)
    interior_mask[1:-1, 1:-1] = True
    
    grid[(interior_mask) & (np.pad(random_roll < wall_density, 1, constant_values=False))] = 0
    grid[(interior_mask) & (np.pad(random_roll > 1 - trap_density, 1, constant_values=False))] = 50
    
    grid[-2, -2] = 200    # Bottom-right goal
    
    return grid


rows, cols = 12, 12
grid_bitmap = generate_enclosed_map(rows, cols)

MOVEMENTS = [(-1, 0), (1, 0), (0, -1), (0,1)]
#               UP  -- DOWN -- LEFT -- RIGHT  

def my_policy(stato):
    return np.random.randint(0,4)
def step(s, action_idx, grid_bitmap, reward_grid, terminal_mask):

    s_next = s + MOVEMENTS[action_idx]
    reward = reward_grid[s_next]
    done = terminal_mask[grid_bitmap[s_next]]

    # implement a step
    return s_next, reward, done

def simulate_episode(start_state, grid_bitmap, reward_grid, terminal_mask, policy, gamma=0.9, max_steps=100):

    # Calculate Return G_t = sum of discounted rewards
    # We calculate backwards for efficiency: G = r + gamma * G

    #use the step
    G_t = []
    
    for i in range(max_steps):
        action = my_policy(i)
        new_state,reward,done = step(start_state,action,grid_bitmap,reward_grid,terminal_mask) 

    # Return starting state and the total discounted return from t=0
    return[-1]



# define rewards
WORLD_CONFIG = {
    0:   {"reward": 0,      "terminal": False, "label": "Wall"},   # Penalty handled by 'stay-put'
    50:  {"reward": -10,    "terminal": True,  "label": "Trap"},   # Big penalty, ends game
    200: {"reward": 10,     "terminal": True,  "label": "Goal"},   # Big prize, ends game
    255: {"reward": -1,     "terminal": False, "label": "Path"}    # Step cost to encourage speed
}

# convert config to two maps like the one we generated 
def parse_grid(grid_bitmap):
    rows, cols = grid_bitmap.shape
    reward_grid = np.zeros((rows, cols))
    terminal_mask = np.zeros((rows, cols), dtype=bool)

    for val, props in WORLD_CONFIG.items():
        # Masking: find all coordinates where the bitmap matches the value
        mask = (grid_bitmap == val)
        reward_grid[mask] = props["reward"]
        terminal_mask[mask] = props["terminal"]
        
    return reward_grid, terminal_mask

reward_grid, terminal_mask = parse_grid(grid_bitmap)




plt.imshow(grid_bitmap, cmap='gray')
plt.title("Enclosed GridWorld with Random Obstacles")
plt.show()