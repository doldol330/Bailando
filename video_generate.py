import os
from tqdm import tqdm
def img2video(expdir, epoch, audio_path=None):
    video_dir = os.path.join(expdir, "videos",f"ep{epoch:06d}")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    image_dir = os.path.join(expdir, "imgs", f"ep{epoch:06d}")


    dance_names = sorted(os.listdir(image_dir))
    audio_dir = "data/aistpp_music_feat_7.5fps"
    
    music_names = sorted(os.listdir(audio_dir))
    
    for dance in tqdm(dance_names, desc='Generating Videos'):
        #pdb.set_trace()
        name = dance.split(".")[0]
        cmd = f"ffmpeg -r 60 -i {image_dir}/{dance}/frame%06d.png -vb 20M -vcodec mpeg4 -y {video_dir}/{name}.mp4 -loglevel quiet"
        # cmd = f"ffmpeg -r 60 -i {image_dir}/{dance}/%05d.png -vb 20M -vcodec qtrle -y {video_dir}/{name}.mov -loglevel quiet"

        os.system(cmd)
        
        name1 = name.replace('cAll', 'c02')

        if 'cAll' in name:
            music_name = name[-9:-5] + '.wav'
        else:
            music_name = name + '.mp3'
            audio_dir = 'extra/'
            music_names = sorted(os.listdir(audio_dir))
        
        if music_name in music_names:
            print('combining audio!')
            audio_dir_ = os.path.join(audio_dir, music_name)
            print(audio_dir_)
            name_w_audio = name + "_audio"
            cmd_audio = f"ffmpeg -i {video_dir}/{name}.mp4 -i {audio_dir_} -map 0:v -map 1:a -c:v copy -shortest -y {video_dir}/{name_w_audio}.mp4 -loglevel quiet"
            os.system(cmd_audio)


for i in range(5, 160, 5):
    epoch = i
    expdir = "/home/kim/Documents/Bailando/Bailando/experiments/actor_critic/vis"
    audio_path = "/home/kim/Documents/Bailando/Bailando/data/aistpp_music_feat_7.5fps"
    img2video(expdir, epoch, audio_path)
