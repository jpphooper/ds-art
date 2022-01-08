import librosa
from librosa.display import waveshow, specshow
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.primitives import generateMaterials


class ExtractMusicFeatures:
    def __init__(self,
                 filepath,
                 duration,
                 offset,
                 sampling_rate,
                 hop_length,
                 n_mfcc):

        self.filepath = filepath
        self.duration = duration
        self.offset = offset
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

        y, sr = librosa.load(
            self.filepath, sr=self.sampling_rate, duration=self.duration, offset=self.offset)

        harm_perc_dict = self._extract_harmonic_percussive(y)
        tempo_beat_frame_dict = self._extract_beats(y, sr)
        mfcc_dict = self._extract_mfcc(y)
        beat_mfcc_delta_dict = self._extract_beat_mfcc_delta(
            mfcc_dict['mfcc'], tempo_beat_frame_dict['beat_frames'])
        chromagram_dict = self._extract_chromagram(harm_perc_dict['y_harm'])
        beat_chroma_dict = self._extract_beat_chroma(chromagram_dict['chromagram'],
                                                     tempo_beat_frame_dict['beat_frames'])
        music_features = {'y': y,
                          'sr': sr,
                          **self._extract_beats(y, sr),
                          **harm_perc_dict,
                          **mfcc_dict,
                          **beat_mfcc_delta_dict,
                          **chromagram_dict,
                          **beat_chroma_dict,
                          **self._extract_beat_features(beat_chroma_dict['beat_chroma'],
                                                        beat_mfcc_delta_dict['beat_mfcc_delta'])
                          }

        self.music_features = music_features

    def _extract_beats(self, y, sr) -> dict:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        return {'tempo': tempo, 'beat_frames': beat_frames}

    def _extract_harmonic_percussive(self, y) -> dict:
        y_harm, y_perc = librosa.effects.hpss(y)
        return {'y_harm': y_harm, 'y_perc': y_perc}

    def _extract_mfcc(self, y):
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sampling_rate,
            hop_length=self.hop_length,
            n_mfcc=self.n_mfcc)
        return {'mfcc': mfcc}

    def _extract_beat_mfcc_delta(self, mfcc, beat_frames) -> dict:
        beat_mfcc_delta = librosa.util.sync(
            np.vstack([mfcc, librosa.feature.delta(mfcc)]), beat_frames)
        return {'beat_mfcc_delta': beat_mfcc_delta}

    def _extract_chromagram(self, y_harm) -> dict:
        chromagram = librosa.feature.chroma_cqt(
            y=y_harm, sr=self.sampling_rate)
        return {'chromagram': chromagram}

    def _extract_beat_chroma(self, chromagram, beat_frames) -> dict:
        beat_chroma = librosa.util.sync(
            chromagram, beat_frames, aggregate=np.median)
        return {'beat_chroma': beat_chroma}

    def _extract_beat_features(self, beat_chroma, beat_mfcc_delta):
        beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
        return {'beat_features': beat_features}

    def visualise_waveshow(self, waveshow_list):
        fig, ax = plt.subplots(nrows=1, figsize=(30, 7))
        if 'Mono' in waveshow_list:
            waveshow(
                self.music_features['y'], sr=self.music_features['sr'], alpha=0.5, ax=ax, color='b', label='Mono')
        if 'Percussive' in waveshow_list:
            waveshow(
                self.music_features['y_perc'], sr=self.music_features['sr'], alpha=0.5, ax=ax, color='r', label='Percussive')
        if 'Harmonic' in waveshow_list:
            waveshow(
                self.music_features['y_harm'], sr=self.music_features['sr'], alpha=0.5, ax=ax, color='g', label='Harmonic')
        ax.set(title='Wave Show')
        ax.label_outer()
        ax.legend()

        return fig

    def visualise_specshow(self, spec_option):
        fig, ax = plt.subplots(nrows=1, figsize=(30, 7))
        if spec_option == 'Chromagram':
            specshow(self.music_features['chromagram'], sr=self.music_features['sr'],
                     hop_length=self.hop_length, cmap='YlOrBr')
            ax.set(title=f'Spec Show - {spec_option}')
            ax.label_outer()
            ax.legend()
        elif spec_option == 'MFCC':
            specshow(self.music_features['mfcc'], sr=self.music_features['sr'],
                     hop_length=self.hop_length, cmap='YlOrBr')
            ax.set(title=f'Spec Show - {spec_option}')
            ax.label_outer()
            ax.legend()
        elif spec_option == 'Beat MFCC Delta':
            specshow(self.music_features['beat_mfcc_delta'], sr=self.music_features['sr'],
                     hop_length=self.hop_length, cmap='YlOrBr')
            ax.set(title=f'Spec Show - {spec_option}')
            ax.label_outer()
            ax.legend()
        elif spec_option == 'Beat Chroma':
            specshow(self.music_features['beat_chroma'], sr=self.music_features['sr'],
                     hop_length=self.hop_length, cmap='YlOrBr')
            ax.set(title=f'Spec Show - {spec_option}')
            ax.label_outer()
            ax.legend()
        elif spec_option == 'Beat Features':
            specshow(self.music_features['beat_features'], sr=self.music_features['sr'],
                     hop_length=self.hop_length, cmap='YlOrBr')
            ax.set(title=f'Spec Show - {spec_option}')
            ax.label_outer()
            ax.legend()

        return fig

    def visualise_tile(self, final_tile_option, size_of_tile):
        fig, ax = plt.subplots(nrows=2, figsize=(30, 7))
        music_feature_options = {
            'Harmonic': self.music_features['y_harm'],
            'Percussive': self.music_features['y_perc'],
            'Mono': self.music_features['y'],
            'Chromagram': self.music_features['chromagram']
        }
        first_arr = music_feature_options[final_tile_option[0]]

        if 'Chromogram' not in final_tile_option:
            if len(final_tile_option) == 2:
                second_arr = music_feature_options[final_tile_option[1]]
            first_matrix, second_matrix = [], []
            for _ in range(len(first_arr[:size_of_tile])):
                first_matrix.append(first_arr[:size_of_tile])
                if len(final_tile_option) == 2:
                    second_matrix.append(second_arr[:size_of_tile])
            tile = np.array(first_matrix)
            if len(final_tile_option) == 2:
                second_tile = np.array(second_matrix)
                tile = np.multiply(100 * tile, 200 * np.transpose(second_tile))

        elif 'Chromagram' in final_tile_option:
            first_arr = music_feature_options['Chromagram'][0]
            final_tile_option.remove('Chromagram')
            first_matrix = []
            for arr in first_arr:
                loop = True
                row = []
                while loop:
                    row.extend(arr)
                    if len(row) > size_of_tile:
                        first_matrix.append(row[:size_of_tile])
                        loop = False
            loop = True
            for row in first_matrix:
                while loop:
                    first_matrix.append(row)
                    if len(first_matrix) > size_of_tile:
                        first_matrix = first_matrix[:size_of_tile]
                        loop = False
            tile = first_matrix
            if len(final_tile_option) == 1:
                second_arr = music_feature_options[final_tile_option[0]]
                second_matrix = []
                for _ in range(len(second_arr[:size_of_tile])):
                    second_matrix.append(second_arr[:size_of_tile])
                second_tile = np.array(second_matrix)
                tile = np.add(tile, 0.5 * np.transpose(second_tile))

        # Set up a figure twice as tall as it is wide
        fig = plt.figure(figsize=plt.figaspect(2.))

        # First subplot
        ax = fig.add_subplot(2, 1, 1)

        ax.set(title='Tile 2D')
        ax.imshow(tile, interpolation='bilinear',
                  norm=colors.Normalize(), cmap='YlOrBr')

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Second subplot
        ax = fig.add_subplot(2, 1, 2, projection='3d')

        ax.set(title='Tile 3D')
        x = np.arange(0, size_of_tile, 1)
        y = np.arange(0, size_of_tile, 1)

        tile = tile - tile.min()

        xs, ys = np.meshgrid(x, y)
        ax.plot_surface(xs, ys, tile)

        return fig, tile


def create_3d_tile():
    sponge = [
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ],
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ],
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
    ]
    model = VoxelModel(sponge, generateMaterials(4))  # 4 is aluminium.
    mesh = Mesh.fromVoxelModel(model)
    return mesh
