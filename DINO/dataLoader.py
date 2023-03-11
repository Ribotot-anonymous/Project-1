import torch, numpy, random, os, math, glob, soundfile
from torch.utils.data import Dataset, DataLoader
from scipy import signal

class train_loader(Dataset):
    def __init__(self, global_frames, train_list, train_path, musan_path, **kwargs):
        self.global_frames = global_frames
        self.data_list = []
        self.noisetypes = ['noise','speech','music'] # Type of noise
        self.noisesnr = {'noise':[0,18],'speech':[3,18],'music':[3,18]} # The range of SNR

        self.noiselist = {} 
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav')) # All noise files in list
        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file) # All noise files in dic
        self.rir_files = numpy.load('rir.npy') # Load the rir file
        for line in open(train_list).read().splitlines():
            filename = os.path.join(train_path, line.split()[1])
            self.data_list.append(filename) # Load the training data list
        
        self.local_frames = kwargs['local_frames']
        self.local_view_num = kwargs['local_view_num']

    def __getitem__(self, index):
        audio_local = []
        audio_global = []
        for i in range(0,self.local_view_num):
            audio_local.append(loadWAV(self.data_list[index], self.local_frames).astype(numpy.float))
        for i in range(0,2):
            audio_global.append(loadWAV(self.data_list[index], self.global_frames).astype(numpy.float))
        audio_local = numpy.concatenate(audio_local,axis=0).astype(numpy.float)
        audio_global = numpy.concatenate(audio_global,axis=0).astype(numpy.float)

        audio_local_aug = []
        audio_global_aug = []

        ## rir list
        rir_filts_list = random.sample(list(self.rir_files), k=2+self.local_view_num)
        ## additive noise list
        noisecat_list  = random.choices(self.noisetypes, k=2+self.local_view_num)
        num_noisecats, noisefiles = [], {}
        for i, noisetype in enumerate(self.noisetypes):
            num_noisecats.append(noisecat_list.count(self.noisetypes[i]))
            noisefiles[noisetype] = random.sample(self.noiselist[self.noisetypes[i]].copy(), k=num_noisecats[i])

        for ii, (rir_filts, noisecat) in enumerate(zip(rir_filts_list, noisecat_list)):
            rir_gains = numpy.random.uniform(-7,3,1)
            noisefile = noisefiles[noisecat].pop()
            snr = [random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])]
            p = random.random()
            if p < 0.15:
                augment_profiles = {'rir_filt':None, 'rir_gain':None, 'add_noise': None, 'add_snr': None}
            elif p < 0.3:
                augment_profiles = {'rir_filt':rir_filts, 'rir_gain':rir_gains, 'add_noise': None, 'add_snr': None}
            elif p < 0.65:
                augment_profiles = {'rir_filt':None, 'rir_gain':None, 'add_noise': noisefile, 'add_snr': snr}
            else:
                augment_profiles = {'rir_filt':rir_filts, 'rir_gain':rir_gains, 'add_noise': noisefile, 'add_snr': snr}

            if ii < 2:
                audio_global_aug.append(self.augment_wav(audio_global[ii], augment_profiles, self.global_frames))
            else:
                audio_local_aug.append(self.augment_wav(audio_local[ii-2], augment_profiles, self.local_frames))

        audio_local_aug = numpy.concatenate(audio_local_aug,axis=0)
        audio_global_aug = numpy.concatenate(audio_global_aug,axis=0)

        with torch.no_grad():
            feat_local = torch.FloatTensor(audio_local_aug)
            feat_global = torch.FloatTensor(audio_global_aug)

        return feat_global, feat_local

    def __len__(self):
        return len(self.data_list)

    def augment_wav(self,audio,augment,fixed_frame):
        if augment['rir_filt'] is not None:
            rir     = numpy.multiply(augment['rir_filt'], pow(10, 0.1 * augment['rir_gain']))
            audio   = signal.convolve(audio, rir, mode='full')[:len(audio)]
        if augment['add_noise'] is not None:
            noiseaudio  = loadWAV(augment['add_noise'], fixed_frame).astype(numpy.float)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
            noise = numpy.sqrt(10 ** ((clean_db - noise_db - augment['add_snr']) / 10)) * noiseaudio
            audio = audio + noise
        else:
            audio = numpy.expand_dims(audio, 0)
        return audio

def loadWAV(filename, max_frames):
    max_audio = max_frames * 160 + 240 # 240 is for padding, for 15ms since window is 25ms and step is 10ms.
    audio, _ = soundfile.read(filename)
    audiosize = audio.shape[0]
    if audiosize <= max_audio: # Padding if the length is not enough
        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
        audio       = numpy.pad(audio, (shortage, shortage), 'wrap')
        audiosize   = audio.shape[0]
    startframe = numpy.int64(random.random()*(audiosize-max_audio)) # Randomly select a start frame to extract audio
    feat = numpy.stack([audio[int(startframe):int(startframe)+max_audio]],axis=0)
    return feat

def get_loader(args): # Define the data loader
    trainLoader = train_loader(**vars(args))
    sampler = torch.utils.data.DistributedSampler(trainLoader, shuffle=True)
    trainLoader = torch.utils.data.DataLoader(
        trainLoader,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.n_cpu,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=5,
    )
    return trainLoader
