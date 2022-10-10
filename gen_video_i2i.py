import os
import math
import numpy as np
import torch
import click
import dnnlib
import legacy
import scipy
import dlib
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import VideoFileClip
from models.fastae_v6_networks import Generator
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image
from aubio import source, tempo


torch.backends.cudnn.benchmark = True
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
])


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def get_beats(in_video: str, fps: int):
    # get beats from audio
    win_s = 512                 # fft size
    hop_s = win_s // 2          # hop size

    try:
        s = source(in_video, 0, hop_s)
    except:
        in_audio, _ = os.path.splitext(in_video)
        in_audio = in_audio + '.wav'
        s = source(in_audio, 0, hop_s)

    samplerate = s.samplerate
    o = tempo("default", win_s, hop_s, samplerate)
    delay = 4. * hop_s
    # list of beats, in samples
    beats = []

    # total number of frames read
    total_frames = 0
    while True:
        samples, read = s()
        is_beat = o(samples)
        if is_beat:
            this_beat = int(total_frames - delay + is_beat[0] * hop_s)
            beats.append(this_beat/ float(samplerate))
        total_frames += read
        if read < hop_s: break
    #print len(beats)
    beats = [math.ceil(i*fps) for i in beats]
    return beats


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--in_video', help='input video file path', type=str, required=True, metavar='FILE')
@click.option('--out_video', help='output video file path', type=str, required=True, metavar='FILE')
@click.option('--mode', help='output video file path', type=click.Choice(['normal', 'blend', 'beat', 'eig']), default='normal')
def video_translation(
    network_pkl: str,
    in_video: str,
    out_video: str,
    mode: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G: Generator = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    latent_dim = G.latent_dim
    G_A2B = G

    clip = VideoFileClip(in_video)
    audioClip = clip.audio

    # Frame numbers and length of output video
    start_frame=0
    end_frame=None
    num_frames = int(clip.fps * clip.duration)
    video_fps= math.ceil(clip.fps)
    faces = None
    smoothing_sec=.7
    eig_dir_idx = 1 # first eig isnt good so we skip it

    frames = []
    beats = get_beats(in_video, video_fps)

    if mode == 'blend':
        shape = [num_frames, 8, latent_dim] # [frame, image, channel, component]
        # TODO
        # all_latents = random_state.randn(*shape).astype(np.float32)
        all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * video_fps, 0, 0], mode='wrap')
        all_latents /= np.sqrt(np.mean(np.square(all_latents)))
        all_latents = torch.from_numpy(all_latents).to(device)
    else:
        all_latents = torch.randn([8, latent_dim]).to(device)
        
    if mode == 'eig':
        # TODO
        all_latents = G_A2B.mapping(all_latents)
        
    in_latent = all_latents

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames

    for frame_num in tqdm(range(num_frames), total=num_frames, desc='frames'):
        image = clip.reader.read_frame()
        if image is None:
            break

        if frame_num < start_frame:
            continue
        # Image size
        height, width = image.shape[:2]

        # 2. Detect with dlib
        if faces is None:
            gray = image.mean(2).astype(np.uint8)
            faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

        # --- Prediction ---------------------------------------------------
        # Face crop with dlib and bounding box scale enlargement
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = image[y:y+size, x:x+size]
        cropped_face = Image.fromarray(cropped_face)
        frame = test_transform(cropped_face).unsqueeze(0).to(device)

        with torch.no_grad():
            A2B_content, A2B_style = G_A2B.encode(frame)
            if isinstance(A2B_content, list):
                A2B_content = list(map(lambda c: c.repeat(8,1,1,1), A2B_content))
            else:
                A2B_content = A2B_content.repeat(8,1,1,1)

            if mode == 'blend':
                in_latent = all_latents[frame_num]
            elif mode == 'normal':
                in_latent = all_latents
            elif mode == 'beat':
                if frame_num in beats:
                    in_latent = torch.randn([8, latent_dim]).to(device)
            
            if mode == 'eig':
                raise NotImplementedError()
                if frame_num in beats:
                    direction = 3 * eigvec[:, eig_dir_idx].unsqueeze(0).expand_as(all_latents).to(device)
                    in_latent = all_latents + direction
                    eig_dir_idx += 1
                    
                fake_A2B = G_A2B.decode(A2B_content, in_latent, use_mapping=False)
            else:
                fake_A2B = G_A2B.decode(A2B_content, in_latent)

            fake_A2B = torch.cat([fake_A2B[:4], frame, fake_A2B[4:]], 0)
            fake_A2B = utils.make_grid(fake_A2B.cpu(), normalize=True, range=(-1, 1), nrow=3)

        #concatenate original image top
        fake_A2B = fake_A2B.permute(1,2,0).cpu().numpy()
        frames.append(fake_A2B*255)
            
    clip = ImageSequenceClip(frames, fps=video_fps)
    clip = clip.set_audio(audioClip)
    clip.write_videofile(out_video)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    video_translation() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------