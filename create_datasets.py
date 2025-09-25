import numpy as np
import slippistats as slp
import peppi_py as peppi
import os
import glob
import pyarrow

def split_combo(combo, last_frame):
    (first, last) = (combo.moves[0].frame_index - 29, min(combo.moves[-1].frame_index + 10, last_frame-30))
    if last <= first:
        return []
    num_samples = np.random.poisson(1/30 * (last - first))
    sample_points = np.random.randint(first, last, size = num_samples)
    samples_and_conts = [(sample, np.array([np.byte(1)]) if sample+30 < combo.moves[-1].frame_index else np.array([np.byte(0)])) for sample in sample_points]
    return samples_and_conts

def extract_character_features(game:peppi.game.Game, sample_time:int, being_comboed:bool, port:int):
    data = game.frames.ports[port].leader
    char = data.post.character.slice(sample_time, 30).to_numpy() #8 bit int
    x_pos = data.post.position.x.slice(sample_time, 30).to_numpy() #float
    y_pos = data.post.position.y.slice(sample_time, 30).to_numpy() #float
    x_vel = data.post.velocities.self_x_ground.slice(sample_time, 30).to_numpy() + data.post.velocities.self_x_air.slice(sample_time, 30).to_numpy() #float
    y_vel = data.post.velocities.self_y.slice(sample_time, 30).to_numpy() #float
    x_knockback = data.post.velocities.knockback_x.slice(sample_time, 30).to_numpy() #float
    y_knockback = data.post.velocities.knockback_y.slice(sample_time, 30).to_numpy() #float
    percent = data.post.percent.slice(sample_time, 30).to_numpy() #float
    hitstun_remaining = data.post.misc_as.slice(sample_time, 30).to_numpy() #float
    hitlag_remaining = data.post.hitlag.slice(sample_time, 30).to_numpy() #float
    direction = data.post.direction.slice(sample_time, 30).to_numpy() #-1 is left, 1 is right, its a float actually
    jumps = data.post.jumps.slice(sample_time, 30).to_numpy() #ints
    action_state = data.post.state.slice(sample_time, 30).to_numpy() #16 bit int, about 400 states, convert to embedding
    action_age = data.post.state_age.slice(sample_time, 30).to_numpy()
    if being_comboed:
        prev_hit = data.post.last_hit_by.slice(sample_time, 30).to_numpy() #93 states, convert to embedding
        combo_count = None
    else:
        prev_hit = None
        combo_count = data.post.combo_count.slice(sample_time, 30).to_numpy()
    return [char, #0
        x_pos, #1
        y_pos,
        x_vel, 
        y_vel, 
        x_knockback, #5
        y_knockback, 
        percent, 
        hitstun_remaining, 
        hitlag_remaining, 
        direction, #10
        jumps, #11
        action_age, #12
        combo_count, #13
        prev_hit, #14
        action_state #15
        ]

def extract_features(game:peppi.game.Game, sample_time:int, comboer_port:int, comboee_port:int):
    sample_time = sample_time + 123 #frame indices are offset by 123
    stage = np.int8(game.start.stage)
    stage = {2:0, 3:1, 8:2, 28:3, 31:4, 32:5}[stage] #convert stage id to range 0-5
    comboer_data = extract_character_features(game, sample_time, False, comboer_port)
    comboee_data = extract_character_features(game, sample_time, True, comboee_port)
    X = np.column_stack([*comboer_data[1:14], *comboee_data[1:13]])
    return {
        "X":X,
        "stage":np.array([stage]*30),
        "comboer state":comboer_data[15],
        "comboer char":comboer_data[0],
        "comboee state":comboee_data[15],
        "comboee char":comboee_data[0],
        "hit list":comboee_data[14]
    }
     
def get_samples(file):
    game = slp.Game(file)
    peppi_game = peppi.read_slippi(file)
    codes = list()
    for i in range(4):
        if game.metadata.players[i] is not None:
            codes.append((i, game.metadata.players[i].connect_code))
            if game.metadata.players[i].connect_code is None:
                return {
                        "X list":list(),
                        "stage list": list(),
                        "comboer state list": list(),
                        "comboer char list":list(),
                        "comboee state list": list(),
                        "comboee char list":list(),
                        "hit list":list(),
                        "yes list": list()
                        }

    comp = slp.ComboComputer()
    comp.prime_replay(game)
    combos = comp.combo_compute(codes[0][1])
    samples_and_conts = list()
    for combo in combos:
        if combo.minimum_length(2):
            samples_and_conts += split_combo(combo, peppi_game.metadata["lastFrame"])
    X_list = list()
    stage_list = list()
    comboer_state_list = list()
    comboer_char_list = list()
    comboee_state_list = list()
    comboee_char_list = list()
    hit_list = list()
    yes_list = list()
    for sample_and_cont in samples_and_conts:
        (sample, yes) = sample_and_cont
        features = extract_features(peppi_game, sample, codes[0][0], codes[1][0])
        X_list.append(features["X"])
        stage_list.append(features["stage"])
        comboer_state_list.append(features["comboer state"])
        comboer_char_list.append(features["comboer char"])
        comboee_state_list.append(features["comboee state"])
        comboee_char_list.append(features["comboee char"])
        hit_list.append(features["hit list"])
        yes_list.append(yes)

    combos = comp.combo_compute(codes[1][1])
    samples_and_conts = list()
    for combo in combos:
        if combo.minimum_length(2):
            samples_and_conts += split_combo(combo, peppi_game.metadata["lastFrame"])
    for sample_and_cont in samples_and_conts:
        (sample, yes) = sample_and_cont
        features = extract_features(peppi_game, sample, codes[1][0], codes[0][0])
        X_list.append(features["X"])
        stage_list.append(features["stage"])
        comboer_state_list.append(features["comboer state"])
        comboer_char_list.append(features["comboer char"])
        comboee_state_list.append(features["comboee state"])
        comboee_char_list.append(features["comboee char"])
        hit_list.append(features["hit list"])
        yes_list.append(yes)
    return {
        "X list":X_list,
        "stage list": stage_list,
        "comboer state list": comboer_state_list,
        "comboer char list":comboer_char_list,
        "comboee state list": comboee_state_list,
        "comboee char list":comboee_char_list,
        "hit list":hit_list,
        "yes list": yes_list
    }

def write_shard(files, shard_index, out_dir):
    X_list = list()
    stage_list = list()
    comboer_state_list = list()
    comboer_char_list = list()
    comboee_state_list = list()
    comboee_char_list = list()
    hit_list = list()
    yes_list = list()
    i = 1
    for file in files:
        if i % 8 == 0:
            print(f"Processed {i:d} files in shard {shard_index:d}")
        samples = get_samples(file)
        X_list.extend(samples["X list"])
        stage_list.extend(samples["stage list"])
        comboer_state_list.extend(samples["comboer state list"])
        comboer_char_list.extend(samples["comboer char list"])
        comboee_state_list.extend(samples["comboee state list"])
        comboee_char_list.extend(samples["comboer char list"])
        hit_list.extend(samples["hit list"])
        yes_list.extend(samples["yes list"])
        i += 1
    X = np.stack(X_list)
    stage = np.stack(stage_list)
    comboer_state = np.stack(comboer_state_list)
    comboer_char = np.stack(comboer_char_list)
    comboee_state = np.stack(comboee_state_list)
    comboee_char = np.stack(comboee_char_list)
    hits = np.stack(hit_list)
    yes = np.stack(yes_list)
    yes = yes.squeeze()
    tmp = os.path.join(out_dir, f"shard_{shard_index:04d}_tmp.npz")
    final = os.path.join(out_dir, f"shard_{shard_index:04d}.npz")
    np.savez(tmp, X=X, stage = stage, comboer_state = comboer_state, comboer_char = comboer_char, 
             comboee_state = comboee_state, comboee_char = comboee_char, hits = hits, y=yes)
    os.replace(tmp, final)

def get_all_samples(dir_path, out_dir):
    files = glob.glob(os.path.join(dir_path, "*.slp"))
    shard_idx = 1
    while 64*shard_idx < len(files):
        write_shard(files[64*(shard_idx-1):(64*shard_idx)], shard_index=shard_idx, out_dir=out_dir)
        print(f"Wrote shard {shard_idx:d}")
        shard_idx += 1
    write_shard(files[64*(shard_idx - 1):], shard_index=shard_idx, out_dir=out_dir)
    print(f"Wrote final shard {shard_idx:d}")
    print("Done")


if __name__ == "__main__":
    print("hello there")
    # get_all_samples(dir_path="file path to slp files", 
    #                  out_dir="file path to folder for training data")
    # get_all_samples(dir_path="file path to slp files", 
    #                out_dir="file path to folder for training data")
    

            


        

