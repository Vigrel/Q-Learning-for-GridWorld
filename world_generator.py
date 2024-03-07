import numpy as np
import randomname


class Cell:
    WALL = "w"
    EMPTY = " "
    AGENT = "a"
    HOLE = "o"
    PRIZE = "g"


def generate_map(size, hole_percentage, prize_percentage, wall_percentage):
    game_map = np.full((size, size), Cell.EMPTY, dtype=str)

    game_map[0, :] = game_map[:, -1] = game_map[:, 0] = game_map[-1, :] = Cell.WALL

    agent_row, agent_col = np.random.randint(1, size - 1, size=2)
    game_map[agent_row, agent_col] = Cell.AGENT

    def place_items(item, percentage):
        num_items = int((size - 2) * (size - 2) * percentage)
        indices = np.random.choice(
            np.arange(1, size - 1), size=(num_items, 2), replace=True
        )
        game_map[indices[:, 0], indices[:, 1]] = item

    place_items(Cell.HOLE, hole_percentage)
    place_items(Cell.PRIZE, prize_percentage)
    place_items(Cell.WALL, wall_percentage)

    return game_map


def save_map(game_map):
    map_name = randomname.get_name().replace("-", "_").upper()
    with open("worlds.py", "a") as file:
        file.write(f"\n{map_name} = ''' \n")
        file.writelines("".join(row) + "\n" for row in game_map)
        file.write("'''\n")


game_map = generate_map(20, 0.05, 0.02, 0.10)
save_map(game_map)
