from src.route import object, tile
import streamlit as st
st.set_page_config(
     page_title="Super Cool App",
     layout="wide",
     menu_items={
         'Get Help': 'https://www.youtube.com/watch?v=eBGIQ7ZuuiU',
         'Report a bug': "https://www.youtube.com/watch?v=eBGIQ7ZuuiU",
         'About': "# Made an app"
     }
 )

PAGES = {
	'Tile':tile
}

def main():
	st.sidebar.title('Navigation')
	selection = st.sidebar.radio("Go to", list(PAGES.keys()))
	page = PAGES[selection]
	page.app()

if __name__ == '__main__':
	
	main()
