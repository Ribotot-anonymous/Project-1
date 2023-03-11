import torch, numpy, random, os, math, glob, soundfile
from torch.utils.data import Dataset, DataLoader
from scipy import signal

class train_loader(Dataset):
    def __init__(self, batch_size, label_ratio, global_frames, local_frames, local_view_num, train_list, train_list_add, train_path, musan_path, **kwargs):
        self.batch_size = batch_size
        self.label_ratio = label_ratio
        self.global_frames = global_frames
        self.local_frames = local_frames
        self.data_list = []
        self.noisetypes = ['noise','speech','music'] # Type of noise
        # self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]} # The range of SNR
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

        with open(train_list_add) as dataset_file:
            lines = dataset_file.readlines();
        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split('-')[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        self.add_data_list = []
        self.add_data_label = []
        self.label2utt_idx = {}

        for index, line in enumerate(lines):
            data = line.strip().split();
            speaker_label = dictkeys[data[0].split('-')[0]];
            filename = os.path.join(train_path,data[1]);
            if not speaker_label in self.label2utt_idx:
                self.label2utt_idx[speaker_label] = []
            self.label2utt_idx[speaker_label].append(filename)

            self.add_data_label.append(speaker_label)
            self.add_data_list.append(filename)

        self.local_view_num = local_view_num

        self.config_suffle_batch()

    def config_suffle_batch(self):
        assert len(self.data_list)//self.label_ratio > len(self.add_data_list)
        #shuffle unlabed data
        unlabed_indexes = numpy.arange(len(self.data_list))
        numpy.random.shuffle(unlabed_indexes)
        unlabed_data = numpy.array(self.data_list)[unlabed_indexes].tolist()
        #shuffle labed data
        labed_indexes = numpy.arange(len(self.add_data_list))
        numpy.random.shuffle(labed_indexes)
        labed_data = numpy.array(self.add_data_list)[labed_indexes].tolist()
        labed_label = numpy.array(self.add_data_label)[labed_indexes].tolist()

        while True: # Genearte each labed data list and label
            labed_data = labed_data + labed_data
            labed_label = labed_label + labed_label
            if len(labed_label) > len(unlabed_data)//self.label_ratio:
                break

        minibatch_crop = self.batch_size//(1+self.label_ratio)
        self.minibatch = []
        start, start_label = 0, 0
        while True: # Genearte each minibatch
            end = min(len(unlabed_data), start + minibatch_crop*self.label_ratio)
            if end == len(unlabed_data):
                self.minibatch.append([unlabed_data[start:end]+unlabed_data[:start + minibatch_crop*self.label_ratio - end], labed_data[start_label:start_label + minibatch_crop], labed_label[start_label:start_label + minibatch_crop]])
                break
            else:
                self.minibatch.append([unlabed_data[start:end], labed_data[start_label:start_label + minibatch_crop], labed_label[start_label:start_label + minibatch_crop]])
            start = end
            start_label = start_label + minibatch_crop

    def __getitem__(self, index):
        data_lists, label_data_lists, data_labels = self.minibatch[index]

        unlabel_local_aug = []
        unlabel_global_aug = []
        for data_name in data_lists:

            audio_local = []
            audio_global = []
            for i in range(0,self.local_view_num):
                audio_local.append(loadWAV(data_name, self.local_frames).astype(numpy.float))
            for i in range(0,2):
                audio_global.append(loadWAV(data_name, self.global_frames).astype(numpy.float))
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

            unlabel_local_aug.append(numpy.expand_dims(audio_local_aug, axis=0))
            unlabel_global_aug.append(numpy.expand_dims(audio_global_aug, axis=0))

        unlabel_local_aug = numpy.concatenate(unlabel_local_aug,axis=0)
        unlabel_global_aug = numpy.concatenate(unlabel_global_aug,axis=0)

        with torch.no_grad():
            unlabel_feat_local = torch.FloatTensor(unlabel_local_aug)
            unlabel_feat_global = torch.FloatTensor(unlabel_global_aug)

        label_local_aug = []
        label_global_aug = []
        for data_name, data_label in zip(label_data_lists, data_labels):
            
            audio_local = []
            audio_global = []
            for i in range(0,self.local_view_num):
                # segment = self.label_aug_wav(data_name, data_label, self.local_frames)
                # audio_local.append(segment.astype(numpy.float))
                audio_local.append(loadWAV(data_name, self.local_frames).astype(numpy.float))
            for i in range(0,2):
                # segment = self.label_aug_wav(data_name, data_label, self.global_frames)
                # audio_global.append(segment.astype(numpy.float))
                audio_global.append(loadWAV(data_name, self.global_frames).astype(numpy.float))
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

            label_local_aug.append(numpy.expand_dims(audio_local_aug, axis=0))
            label_global_aug.append(numpy.expand_dims(audio_global_aug, axis=0))

        label_local_aug = numpy.concatenate(label_local_aug,axis=0)
        label_global_aug = numpy.concatenate(label_global_aug,axis=0)

        with torch.no_grad():
            label_feat_local = torch.FloatTensor(label_local_aug)
            label_feat_global = torch.FloatTensor(label_global_aug)

        return unlabel_feat_global, unlabel_feat_local, label_feat_global, label_feat_local, torch.LongTensor(data_labels)

    def __len__(self):
        return len(self.minibatch)

    def label_aug_wav(self, data_name, data_label, max_frames):
        k = random.random()
        if k < 0.8 :
            aug_frame2 = random.randint(20, max_frames//2)
            aug_frame1 = max_frames - aug_frame2
            segment_num1 = loadWAV(data_name, aug_frame1)

            aug_data = random.choice(self.label2utt_idx[data_label])

            segment_num2 = loadWAV(aug_data, aug_frame2)
            segment = numpy.concatenate([segment_num1, segment_num2],axis=-1)[:,120:-120]
        else :
            segment = loadWAV(data_name, max_frames)
        return segment

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
    audio, fs = soundfile.read(filename)
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
    trainLoader = torch.utils.data.DataLoader(
        trainLoader,
        batch_size=1,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=5,
    )
    return trainLoader
