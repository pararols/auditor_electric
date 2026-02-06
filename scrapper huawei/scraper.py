import os
import time
import datetime
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

# Load credentials
load_dotenv()
USERNAME = os.getenv("HUAWEI_USER")
PASSWORD = os.getenv("HUAWEI_PASS")

def run(start_date_arg=None, end_date_arg=None):
    with sync_playwright() as p:
        # Launch browser in headed mode to be visible to the user
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("Navigating to Huawei FusionSolar...")
        # ... (Navigation logic remains same) ...
        # (Assuming we skip to the date loop part to keep this edit valid/concise, 
        # but I need to make sure 'run' signature change propagates if called elsewhere)
        # Actually, let's just parse args inside run or Main.
        
        # ... [Keep existing navigation code] ...
        # Instead of replacing everything, I will replace the wrapper and date definitions.
        
        # ...
        # Launch browser in headed mode to be visible to the user
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("Navigating to Huawei FusionSolar...")
        page.goto("https://eu5.fusionsolar.huawei.com")

        print("Waiting for page load...")
        # page.wait_for_load_state("networkidle") # Can hang on polling sites
        time.sleep(15) # Wait for JS rendering explicitly

        # Save HTML for inspection
        with open("debug_login.html", "w", encoding="utf-8") as f:
            f.write(page.content())
        print("Saved debug_login.html")

        # Wait for login form
        print("Waiting for login form...")
        
        try:
            # Handle Cookie Banner if present
            try:
                if page.is_visible("div#cookie-policy i"):
                    print("closing cookie banner")
                    page.click("div#cookie-policy i")
                    time.sleep(1)
            except Exception as e:
                print(f"Cookie banner error (ignoring): {e}")

            # Type credentials using keyboard to trigger React events
            page.click("#username")
            page.keyboard.type(USERNAME, delay=50)
            time.sleep(0.5)
            
            page.click("#value")
            page.keyboard.type(PASSWORD, delay=50)
            time.sleep(1)
            
            # Try submitting with Enter first
            print("Pressing Enter to login...")
            page.keyboard.press("Enter")
            
            # Wait a bit to see if Enter worked
            time.sleep(3)
            
            if "LOGIN" in page.url or "login.action" in page.url:
                print("Enter didn't navigate, trying click...")
                # Click login button
                if page.is_visible("#btn_outerverify"):
                    page.click("#btn_outerverify")
                elif page.is_visible("#submitDataverify"):
                    page.click("#submitDataverify")
            
            print("Login submitted. Waiting for navigation...")
            
            # Polling for URL change
            for i in range(20):
                if "LOGIN" not in page.url and "login.action" not in page.url:
                    print(f"Login successful! New URL: {page.url}")
                    break
                print(f"Waiting for redirect... ({i+1}/20)")
                time.sleep(2)
                
            print(f"Current URL: {page.url}")

            # Check if still on login page
            if "LOGIN" in page.url or "login.action" in page.url:
                 print("Warning: Still on login page.")

            # Check if still on login page
            if "LOGIN" in page.url or "login.action" in page.url:
                 print("Warning: Still on login page.")
                 # Check for error message
                 if page.is_visible("#errorMessage"):
                     err = page.text_content("#errorMessage")
                     print(f"Login Error Message: {err}")
            
            # Wait a bit more for SPA rendering
            time.sleep(10)
            
            # Snapshot for debugging
            with open("debug_dashboard.html", "w", encoding="utf-8") as f:
                f.write(page.content())
            print("Saved debug_dashboard.html")

            # Robust Station Selection / Auto-redirect handling
            print("Waiting for Station View or Station List (polling)...")
            in_station_view = False
            
            # Poll for 60 seconds
            start_time = time.time()
            while time.time() - start_time < 60:
                # 1. Check if URL is already station view
                if "/view/station/" in page.url or "overview" in page.url:
                    print("Detected Station View via URL (Auto-redirect).")
                    in_station_view = True
                    break
                
                # 2. Check for Station List selectors in Main Page
                if page.is_visible(".ant-table-row a") or page.is_visible(".nco-home-card"):
                    print("Station List detected in Main Page.")
                    break
                    
                # 3. Check for Station List in Frames
                found_in_frame = False
                for frame in page.frames:
                    try:
                        if frame.is_visible(".ant-table-row a") or frame.is_visible(".nco-home-card"):
                            print(f"Station List detected in Frame '{frame.name}'.")
                            found_in_frame = True
                            break
                    except: pass
                if found_in_frame: break
                
                time.sleep(2)
            
            # After polling, decide what to do
            if in_station_view:
                pass # Ready to proceed
            else:
                # Try to click station if found
                print("Trying to click station in list...")
                found_click = False
                
                # Try main page
                stations = page.query_selector_all(".ant-table-row a")
                if not stations: stations = page.query_selector_all(".nco-home-card")
                if stations:
                    print(f"Clicking first station (Main Page): {stations[0].text_content().strip()}")
                    stations[0].click()
                    found_click = True
                
                # Try frames
                if not found_click:
                    for frame in page.frames:
                        try:
                            stations = frame.query_selector_all(".ant-table-row a")
                            if not stations: stations = frame.query_selector_all(".nco-home-card")
                            if stations:
                                print(f"Clicking first station (Frame {frame.name}): {stations[0].text_content().strip()}")
                                stations[0].click()
                                found_click = True
                                break
                        except: pass
                
                if found_click:
                    print("Clicked station. Waiting for navigation...")
                    time.sleep(15)
                    in_station_view = True
                else:
                    if "/view/station/" in page.url:
                        in_station_view = True
                    else:
                        print("Failed to find station and no auto-redirect happened.")

            if in_station_view:
                print(f"Current URL: {page.url}")
                
                # Save station dashboard (main frame)
                with open("debug_station_main.html", "w", encoding="utf-8") as f:
                    f.write(page.content())
                print("Saved debug_station_main.html")
                
                # Save all frames
                print(f"Frames found: {len(page.frames)}")
                for i, frame in enumerate(page.frames):
                    print(f"Frame {i}: Name='{frame.name}', URL='{frame.url}'")
                    try:
                        with open(f"debug_frame_{i}.html", "w", encoding="utf-8") as f:
                            f.write(frame.content())
                        print(f"Saved debug_frame_{i}.html")
                    except Exception as e:
                        print(f"Error saving frame {i}: {e}")

                # Now look for Trend/Historical Data or Device Management
                print("Looking for 'Trend', 'Historical', or 'Device Management' (and Spanish equivalents)...")
                
                target_clicked = False
                
                # Try explicit text matches including Spanish
                targets = ["Trend", "Tendencia", "Historical Data", "Datos Históricos", "Device Management", "Report Management", "Reports", "Informes", "Energy Management", "Gestión de energía"]
                
                # Try finding by title attribute for Tendencia
                if not target_clicked:
                    try:
                        if page.is_visible("[title='Tendencia']"):
                            print("Found element with title='Tendencia'! Clicking...")
                            page.click("[title='Tendencia']")
                            target_clicked = True
                    except: pass

                if not target_clicked:
                    for target in targets:
                        if page.is_visible(f"text={target}"):
                            print(f"Found '{target}'! Clicking...")
                            page.click(f"text={target}")
                            target_clicked = True
                            break
                
                # Check frames for targets
                if not target_clicked:
                    print("Checking frames for targets...")
                    for frame in page.frames:
                        # Try title in frame
                        try:
                            if frame.is_visible("[title='Tendencia']"):
                                print(f"Found title='Tendencia' in frame '{frame.name}'! Clicking...")
                                frame.click("[title='Tendencia']")
                                target_clicked = True
                                break
                        except: pass
                        
                        if target_clicked: break

                        for target in targets:
                            try:
                                # Use exact text strictness=False
                                if frame.is_visible(f"text={target}"):
                                    print(f"Found '{target}' in frame '{frame.name}'! Clicking...")
                                    frame.click(f"text={target}")
                                    target_clicked = True
                                    break
                            except: pass
                        if target_clicked: break
                
                if not target_clicked:
                    pass # Already checked frames
                
                if target_clicked:
                    print("Clicked target. Waiting for page load...")
                    time.sleep(10)
                    with open("debug_target.html", "w", encoding="utf-8") as f:
                         f.write(page.content())
                    print("Saved debug_target.html")
                    
                    # Logic for Exporting Data
                    print("Attempting to select Day view and Export...")
                    
                    # Logic for Exporting Data
                    print("Attempting to select Day view and Iterate Dates...")
                    
                    # 1. Select "Día" (Day)
                    try:
                        if page.is_visible("text=Día"):
                            print("Found 'Día' selector. Clicking...")
                            page.click("text=Día")
                            time.sleep(5)
                    except Exception as e:
                        print(f"Error selecting Day: {e}")

                    # Define Date Range
                    if start_date_arg:
                        start_date = datetime.datetime.strptime(start_date_arg, "%Y-%m-%d").date()
                    else:
                        start_date = datetime.date(2024, 12, 4)
                        
                    if end_date_arg:
                        end_date = datetime.datetime.strptime(end_date_arg, "%Y-%m-%d").date()
                    else:
                        end_date = datetime.date.today()
                        
                    current_date = start_date

                    if not os.path.exists("data"):
                        os.makedirs("data")

                    while current_date <= end_date:
                        # ... (existing loop body) ...
                        date_str = current_date.strftime("%Y-%m-%d")
                        output_file = os.path.abspath(f"data/huawei_hourly_{date_str}.xlsx")
                        
                        if os.path.exists(output_file):
                            print(f"Skipping {date_str}, file exists.")
                            current_date += datetime.timedelta(days=1)
                            continue

                        print(f"Processing date: {date_str}...")
                        
                        try:
                            # Find Date Input
                            date_input = None
                            
                            # Specific selector for "Seleccionar fecha"
                            date_input = page.query_selector("input[placeholder='Seleccionar fecha']")
                            if not date_input:
                                # Fallback to generic search
                                inputs = page.query_selector_all("input")
                                for inp in inputs:
                                     ph = (inp.get_attribute("placeholder") or "").lower()
                                     if "fecha" in ph or "date" in ph:
                                         date_input = inp
                                         break
                            
                            if date_input:
                                # Clear and Fill
                                date_input.click()
                                date_input.fill(date_str)
                                page.keyboard.press("Enter")
                                # Wait for data loading spinner or chart update
                                print("Date set. Waiting for chart update...")
                                time.sleep(5) 
                                
                                # Export
                                print("Clicking 'Exportar'...")
                                with page.expect_download(timeout=60000) as download_info:
                                    if page.is_visible("text=Exportar"):
                                        page.click("text=Exportar")
                                    elif page.is_visible("[title='Exportar']"):
                                        page.click("[title='Exportar']")
                                    else:
                                        btn = page.query_selector("button:has-text('Exportar')")
                                        if btn: btn.click()
                                
                                download = download_info.value
                                print(f"Download triggered. Saving to {output_file}")
                                download.save_as(output_file)
                                print("Download successful.")
                                
                            else:
                                print("Date input not found. Aborting loop.")
                                break

                        except Exception as e:
                            print(f"Error processing {date_str}: {e}")
                            # Don't break, maybe retry or skip?
                            # For now, continue
                        
                        current_date += datetime.timedelta(days=1)
                        # Small pause to avoid rate limiting?
                        time.sleep(2)
                        
                else:
                    print("No target found. Saving snapshot.")
                
            time.sleep(5)
            
        except Exception as e:
            print(f"Error during interaction: {e}")
            try:
                with open("debug_error.html", "w", encoding="utf-8") as f:
                    f.write(page.content())
                print("Saved debug_error.html")
            except:
                pass
            time.sleep(5)
        
        browser.close()

if __name__ == "__main__":
    import argparse
    
    if not USERNAME or not PASSWORD:
        print("Error: Credentials not found in .env file")
    else:
        parser = argparse.ArgumentParser(description="Huawei Solar Scraper")
        parser.add_argument("--start_date", type=str, help="Start date (YYYY-MM-DD)")
        parser.add_argument("--end_date", type=str, help="End date (YYYY-MM-DD)")
        args = parser.parse_args()
        
        run(args.start_date, args.end_date)
