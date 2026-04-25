import time
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from config import (
    MVP_HTML_DIR, PLAYER_HTML_DIR, TEAM_HTML_DIR,
    PLAYERS_RAW_FILE, TEAMS_RAW_FILE, MVPS_RAW_FILE,
    YEARS, ensure_directories
)

def setup_chrome_driver():
    """Setup undetected Chrome driver to bypass Cloudflare"""
    options = uc.ChromeOptions()
    # Note: undetected-chromedriver works best without headless mode
    # headless mode often triggers bot detection
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--window-size=1920,1080')

    # Use undetected-chromedriver to bypass bot detection
    # Specify Chrome version 145 (current installed version)
    driver = uc.Chrome(options=options, version_main=145, use_subprocess=True)
    return driver

def scrape_mvp_data(years):
    """Scrape MVP voting data for given years using Selenium"""
    print("Scraping MVP data with Selenium...")

    driver = setup_chrome_driver()
    url_start = "https://www.basketball-reference.com/awards/awards_{}.html"

    try:
        for year in years:
            url = url_start.format(year)
            print(f"Scraping MVP data for {year}...")

            try:
                driver.get(url)
                time.sleep(3)  # Wait for page to load and Cloudflare check

                with open(MVP_HTML_DIR / f"{year}.html", "w+", encoding='utf-8') as f:
                    f.write(driver.page_source)
                print(f"✓ MVP data for {year} saved")
                time.sleep(2)  # Polite delay between requests
            except Exception as e:
                print(f"✗ Error scraping MVP data for {year}: {e}")
    finally:
        driver.quit()

def scrape_player_data_with_selenium(years):
    """Scrape player statistics using Selenium (needed for full page loading)"""
    print("Scraping player data with Selenium...")

    driver = setup_chrome_driver()
    player_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"

    try:
        for year in years:
            url = player_stats_url.format(year)
            print(f"Scraping player data for {year}...")

            try:
                driver.get(url)
                time.sleep(3)  # Wait for Cloudflare check
                driver.execute_script("window.scrollTo(1,10000)")
                time.sleep(2)  # Wait for dynamic content to load

                with open(PLAYER_HTML_DIR / f"{year}.html", "w+", encoding='utf-8') as f:
                    f.write(driver.page_source)
                print(f"✓ Player data for {year} saved")
                time.sleep(2)  # Polite delay between requests
            except Exception as e:
                print(f"✗ Error scraping player data for {year}: {e}")
    finally:
        driver.quit()

def scrape_team_data(years):
    """Scrape team standings data using Selenium"""
    print("Scraping team data with Selenium...")

    driver = setup_chrome_driver()
    team_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_standings.html"

    try:
        for year in years:
            url = team_stats_url.format(year)
            print(f"Scraping team data for {year}...")

            try:
                driver.get(url)
                time.sleep(3)  # Wait for page to load and Cloudflare check

                with open(TEAM_HTML_DIR / f"{year}.html", "w+", encoding='utf-8') as f:
                    f.write(driver.page_source)
                print(f"✓ Team data for {year} saved")
                time.sleep(2)  # Polite delay between requests
            except Exception as e:
                print(f"✗ Error scraping team data for {year}: {e}")
    finally:
        driver.quit()

def process_mvp_data(years):
    """Process scraped MVP HTML files into DataFrame"""
    print("Processing MVP data...")
    dfs = []

    for year in years:
        try:
            with open(MVP_HTML_DIR / f"{year}.html", encoding='utf-8') as f:
                page = f.read()

            soup = BeautifulSoup(page, 'html.parser')

            # Remove header row that interferes with parsing
            over_header = soup.find('tr', class_="over_header")
            if over_header:
                over_header.decompose()

            mvp_table = soup.find_all(id="mvp")[0]
            mvp_df = pd.read_html(str(mvp_table))[0]
            mvp_df["Year"] = year
            dfs.append(mvp_df)
            print(f"✓ Processed MVP data for {year}")
        except Exception as e:
            print(f"✗ Error processing MVP data for {year}: {e}")

    return pd.concat(dfs) if dfs else pd.DataFrame()

def process_player_data(years):
    """Process scraped player HTML files into DataFrame"""
    print("Processing player data...")
    dfs = []

    for year in years:
        try:
            with open(PLAYER_HTML_DIR / f"{year}.html", encoding='utf-8') as f:
                page = f.read()

            soup = BeautifulSoup(page, 'html.parser')

            # Remove header rows that interfere with parsing
            for thead in soup.find_all('tr', class_="thead"):
                thead.decompose()

            player_table = soup.find_all(id="per_game_stats")[0]
            player_df = pd.read_html(str(player_table))[0]
            player_df["Year"] = year
            dfs.append(player_df)
            print(f"✓ Processed player data for {year}")
        except Exception as e:
            print(f"✗ Error processing player data for {year}: {e}")

    return pd.concat(dfs) if dfs else pd.DataFrame()

def process_team_data(years):
    """Process scraped team HTML files into DataFrame"""
    print("Processing team data...")
    dfs = []

    for year in years:
        try:
            with open(TEAM_HTML_DIR / f"{year}.html", encoding='utf-8') as f:
                page = f.read()

            soup = BeautifulSoup(page, 'html.parser')

            # Remove header rows that interfere with parsing
            for thead in soup.find_all('tr', class_="thead"):
                thead.decompose()

            # Process Eastern Conference
            e_table = soup.find_all(id="divs_standings_E")[0]
            e_df = pd.read_html(str(e_table))[0]
            e_df["Year"] = year
            e_df["Team"] = e_df["Eastern Conference"]
            del e_df["Eastern Conference"]
            dfs.append(e_df)

            # Process Western Conference
            w_table = soup.find_all(id="divs_standings_W")[0]
            w_df = pd.read_html(str(w_table))[0]
            w_df["Year"] = year
            w_df["Team"] = w_df["Western Conference"]
            del w_df["Western Conference"]
            dfs.append(w_df)

            print(f"✓ Processed team data for {year}")
        except Exception as e:
            print(f"✗ Error processing team data for {year}: {e}")

    return pd.concat(dfs) if dfs else pd.DataFrame()

def main():
    print(f"Scraping data for years: {min(YEARS)} to {max(YEARS)}")
    print("Using automatic ChromeDriver management")

    # Create directories
    ensure_directories()

    # Scrape data - all using Selenium to avoid 403 errors
    print("\n💡 Using Selenium for all scraping to bypass website restrictions\n")
    scrape_mvp_data(YEARS)
    scrape_team_data(YEARS)
    scrape_player_data_with_selenium(YEARS)

    # Process scraped data into DataFrames and save as CSV
    print("\n" + "="*50)
    print("PROCESSING DATA")
    print("="*50)

    mvps = process_mvp_data(YEARS)
    if not mvps.empty:
        mvps.to_csv(MVPS_RAW_FILE, index=False)
        print(f"MVP data saved to {MVPS_RAW_FILE}")

    players = process_player_data(YEARS)
    if not players.empty:
        players.to_csv(PLAYERS_RAW_FILE, index=False)
        print(f"Player data saved to {PLAYERS_RAW_FILE}")

    teams = process_team_data(YEARS)
    if not teams.empty:
        teams.to_csv(TEAMS_RAW_FILE, index=False)
        print(f"Team data saved to {TEAMS_RAW_FILE}")

    print("\n" + "="*50)
    print("DATA SCRAPING COMPLETED!")
    print("="*50)
    print(f"Years scraped: {min(YEARS)} to {max(YEARS)}")
    print(f"Files created: {MVPS_RAW_FILE.name}, {PLAYERS_RAW_FILE.name}, {TEAMS_RAW_FILE.name}")

if __name__ == "__main__":
    main()
