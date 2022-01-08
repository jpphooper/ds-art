import streamlit as st
import tempfile
from stl import mesh
from pathlib import Path
from ..utils import sound


def app():
    st.title('Data Art - Create a Tile')

    uploaded_file = st.file_uploader("Choose your .wav file", type=['.wav'])

    duration = st.sidebar.slider('Sound Duration (Seconds)',
                                 min_value=1,
                                 max_value=30,
                                 value=10,
                                 step=1)

    offset = st.sidebar.slider('Start Sound From (Seconds)',
                               min_value=0,
                               max_value=100,
                               value=0,
                               step=1)

    sampling_rate = st.sidebar.slider('Sound Sampling Rate',
                                      min_value=1000,
                                      max_value=220500,
                                      value=100000,
                                      step=500)

    hop_length = st.sidebar.slider('Hop Length',
                                   min_value=1,
                                   max_value=100,
                                   value=10,
                                   step=1)

    n_mfcc = st.sidebar.slider('N MFCC',
                               min_value=1,
                               max_value=100,
                               value=10,
                               step=1)

    size_of_tile = st.sidebar.slider('Size of Tile',
                                     min_value=100,
                                     max_value=1000,
                                     value=250,
                                     step=50)

    wave_options = st.sidebar.multiselect('Choose Waves',
                                          ['Mono', 'Harmonic', 'Percussive'],
                                          ['Harmonic', 'Percussive'])

    spec_option = st.sidebar.selectbox('Choose Spectrogram',
                                       ['Chromagram', 'MFCC', 'Beat MFCC Delta', 'Beat Chroma', 'Beat Features'])

    final_tile_option = st.sidebar.multiselect('Choose Final Tile',
                                               ['Harmonic', 'Percussive',
                                                   'Mono', 'Chromagram'],
                                               ['Harmonic', 'Percussive'])

    if len(final_tile_option) > 2:
        st.sidebar.error('Can only choose up to 2 final tile options')

    if uploaded_file is not None:
        st.text('Play full audio file:')
        st.audio(uploaded_file)

        st.text('Play audio from chosen start time')
        st.audio(uploaded_file, start_time=offset)

        # Make temp file path from uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:

            fp = Path(tmp_file.name)
            fp.write_bytes(uploaded_file.getvalue())

        MusicExtractor = sound.ExtractMusicFeatures(filepath=fp,
                                                    duration=duration,
                                                    offset=offset,
                                                    sampling_rate=sampling_rate,
                                                    hop_length=hop_length,
                                                    n_mfcc=n_mfcc)

        st.pyplot(MusicExtractor.visualise_waveshow(wave_options))

        st.pyplot(MusicExtractor.visualise_specshow(spec_option))

        fig, tile = MusicExtractor.visualise_tile(
            final_tile_option, size_of_tile)

        st.pyplot(fig)

        tile_mesh = sound.create_3d_tile()

        tile_mesh.export('your_tile.stl')

        st.sidebar.download_button('Download STL file', 'your_tile.stl', file_name='your_tile.stl')
