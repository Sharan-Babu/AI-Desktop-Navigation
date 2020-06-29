from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pyautogui

class YouTube():
    
    def __init__(self):
        self.screen = None
        self.driver = webdriver.Chrome(r'C:\Users\Mat\Desktop\chromedriver_win32\chromedriver')

    def go_to(self):
        self.driver.get('https://www.youtube.com/')
        self.driver.maximize_window()
        
    def click_trending(self):
        trending = self.driver.find_element_by_css_selector(r"[href='\/feed\/trending'] .title")
        trending.click()

    def close_youtube(self):
        self.driver.close()

    def previous_page(self):
        self.driver.back()

    def next_page(self):
        self.driver.forward()        	

    def create(self):
        trending = self.driver.find_element_by_xpath(r"/html/body/ytd-app/div/div/ytd-masthead/div[3]/div[3]/div[2]/ytd-topbar-menu-button-renderer[1]/div/a/yt-icon-button/button/yt-icon")
        trending.click()

    def next_item(self):
            pyautogui.press('tab')

    def sign_in(self):
        sign = self.driver.find_element_by_xpath(r"/html/body/ytd-app/div/div/ytd-masthead/div[3]/div[3]/div[2]/ytd-button-renderer/a/paper-button/yt-icon")
        sign.click()    

    def select_first_account(self):
            pyautogui.press('tab')
            pyautogui.press('enter')

    def select_second_account(self):
            pyautogui.press('tab')
            pyautogui.press('tab')
            pyautogui.press('enter')

    def write_password(self, password):
            pyautogui.write(password)
            pyautogui.press('tab')
            pyautogui.press('tab')
            pyautogui.press('enter')

    def pause_or_play(self):
        pyautogui.write(['space'])

    def skip_5_seconds(self):
            pyautogui.write(['right'])

    def rewind_5_seconds(self):
        pyautogui.write(['left'])


    def sign_out(self):
            elem = self.driver.find_element_by_xpath(r'//*[@id="avatar-btn"]')
            elem.click()
            for i in range(6):
                    pyautogui.press('tab')
            pyautogui.press('enter')


    def toggle_autoplay(self):
        auto = self.driver.find_element_by_xpath(r"/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[2]/div/div[3]/ytd-watch-next-secondary-results-renderer/div[2]/ytd-compact-autoplay-renderer/div[1]/paper-toggle-button/div[1]")
        auto.click()

    def like_video(self):
        like = self.driver.find_element_by_xpath(r"/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[1]/div/div[5]/div[2]/ytd-video-primary-info-renderer/div/div/div[3]/div/ytd-menu-renderer/div/ytd-toggle-button-renderer[1]/a")
        like.click()

    def dislike_video(self):
        dislike = self.driver.find_element_by_xpath(r"/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[1]/div/div[5]/div[2]/ytd-video-primary-info-renderer/div/div/div[3]/div/ytd-menu-renderer/div/ytd-toggle-button-renderer[2]/a")
        dislike.click()

    def search(self, msg):
        qsearch = self.driver.find_element_by_xpath(r'/html/body/ytd-app/div/div/ytd-masthead/div[3]/div[2]/ytd-searchbox/form/div/div[1]/input')
        qsearch.send_keys(msg)
        qbutton = self.driver.find_element_by_xpath(r'/html/body/ytd-app/div/div/ytd-masthead/div[3]/div[2]/ytd-searchbox/form/button')
        qbutton.click()
        
    def theatre_mode(self):
        self.screen='theatre'
        pyautogui.write(['t'])	

    def increase_speed(self):
        pyautogui.hotkey('shift','.') 

    def decrease_speed(self):
        pyautogui.hotkey('shift',',')        

    def zoom_out(self):
            if self.screen=='theatre':
                    pyautogui.write(['t'])
            else:
                pyautogui.write(['f'])	

    def mute_unmute(self):
            pyautogui.write(['m'])

    def skip_to_section(self, num):
            pyautogui.write([num])

    def next_video(self):
            pyautogui.hotkey('shift','n')

    def previous_video(self):
            pyautogui.hotkey('shift','p')

    def full_screen(self):
        self.screen
        screen='full_screen'
        pyautogui.write(['f'])	

    def play_first_video(self):
            time.sleep(3)
            pyautogui.press('tab')
            pyautogui.press('enter')

    def play_second_video(self):
            time.sleep(3)
            for i in range(4):
                    pyautogui.press('tab')
            pyautogui.press('enter')	

    def play_third_video(self):
        time.sleep(3)
        for i in range(7):
            pyautogui.press('tab')
        pyautogui.press('enter')       	
