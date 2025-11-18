import nflreadpy as nfl

pbp = nfl.load_pbp()

player_stats = nfl.load_player_stats([2025])

print(player_stats[0].player_name)



