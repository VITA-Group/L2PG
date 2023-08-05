import numpy as np

"""
with open("game_list.txt", "w") as f:
    n_player = 2
    for i in range(20):
        a = np.random.randn(n_player ** 2 + n_player)
        w, v = np.linalg.eig(np.array([a[0], a[2], a[3], a[1]]).reshape(n_player, n_player))
        print(w)
        while w[0] < 0 or w[1] < 0:
            a = np.random.randn(n_player ** 2 + n_player)
            w, v = np.linalg.eig(np.array([a[0], a[2], a[3], a[1]]).reshape(n_player, n_player))
            print(w)
        f.write(",".join(map(str, list(a))) + "\n")
"""

def generate_stable_games(save_file=True, mode="uniform", n_player=2, num_games=20):
    if mode == "uniform":
        rng = np.random.rand
    else:
        rng = np.random.randn
    games = []
    for _ in range(num_games):
        a = rng(n_player ** 2 + n_player)
        w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(n_player, n_player))

        while w[0] < 0 or w[1] < 0:
            a = rng(n_player ** 2 + n_player)
            w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(n_player, n_player))
        games.append(a)

    if save_file:
        with open(f"stable_game_list_{mode}.txt", "w") as f:
            for a in games:
                f.write(",".join(map(str, list(a))) + "\n")

    return games

def generate_unstable_games(save_file=True, mode="uniform", n_player=2, num_games=20):
    if mode == "uniform":
        rng = np.random.rand
    else:
        rng = np.random.randn
    games = []
    for _ in range(num_games):
        a = rng(n_player ** 2 + n_player)
        w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(n_player, n_player))

        while w[0] > 0 or w[1] > 0:
            a = rng(n_player ** 2 + n_player)
            w, v = np.linalg.eig(np.array([a[0], (a[2] + a[3]) / 2, (a[2] + a[3]) / 2, a[1]]).reshape(n_player, n_player))
        games.append(a)

    if save_file:
        with open(f"unstable_game_list_{mode}.txt", "w") as f:
            for a in games:
                f.write(",".join(map(str, list(a))) + "\n")

    return games

if __name__ == "__main__":
    # generate_stable_games(True, mode='normal')
    generate_unstable_games(True, mode='normal')
