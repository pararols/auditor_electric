import requests
import json
import time

class HuaweiClient:
    def __init__(self, username, password, base_url="https://eu5.fusionsolar.huawei.com"):
        self.username = username
        self.password = password
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        })

    def login(self):
        """
        Logs in to the Northbound API.
        Endpoint: /thirdData/login
        """
        url = f"{self.base_url}/thirdData/login"
        payload = {
            "userName": self.username,
            "systemCode": self.password
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=10)
            data = response.json()
            
            if data.get("success", False):
                self.token = response.headers.get("XSRF-TOKEN")
                # Sometimes token is in header, sometimes in body depending on version. 
                # Northbound API usually requires xsrf-token in header for subsequent requests.
                if not self.token:
                    # Try to get from cookies or data
                    self.token = response.cookies.get("XSRF-TOKEN")
                
                if self.token:
                    self.session.headers.update({"XSRF-TOKEN": self.token})
                    return True
                else:
                    # Some versions return specific token in body
                    if "data" in data and "requestToken" in data["data"]:
                         self.token = data["data"]["requestToken"] # older version?
                         self.session.headers.update({"XSRF-TOKEN": self.token})
                         return True

            print(f"Login Failed: {data}")
            return False
        except Exception as e:
            print(f"Login Error: {e}")
            return False

    def get_station_list(self):
        """
        Retrieves the list of stations.
        Endpoint: /thirdData/getStationList
        """
        if not self.token:
            if not self.login(): return None

        url = f"{self.base_url}/thirdData/getStationList"
        # Body usually allows filtering, empty for all
        payload = {
            "pageNo": 1,
            "pageSize": 100
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=10)
            data = response.json()
            if data.get("success", False):
                return data.get("data", {}).get("list", [])
            else:
                # Token might be expired
                if data.get("failCode") == 305 or data.get("failCode") == 401:
                    print("Token expired, retrying login...")
                    if self.login():
                        return self.get_station_list()
                print(f"Get Station List Failed: {data}")
                return None
        except Exception as e:
            print(f"Error fetching stations: {e}")
            return None

    def get_kpi_station_hour(self, station_codes, collect_time):
        """
        Retrieves hourly KPI data for stations.
        Endpoint: /thirdData/getKpiStationHour
        collect_time: Timestamp (long) or string? API doc says Long (milliseconds).
        """
        if not self.token:
            if not self.login(): return None
            
        url = f"{self.base_url}/thirdData/getKpiStationHour"
        payload = {
            "stationCodes": ",".join(station_codes) if isinstance(station_codes, list) else station_codes,
            "collectTime": collect_time # Long: Milliseconds
        }
        
        # Retry logic internal to Client
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # print(f"DEBUG REQ: {url} payload={payload}")
                response = self.session.post(url, json=payload, timeout=10)
                data = response.json()
                # print(f"DEBUG RESP: {data}")
                
                if data.get("success", False):
                    return data.get("data", [])
                
                # Check 407
                if data.get("failCode") == 407:
                    # Exponential Backoff
                    wait = 5 * (attempt + 1)
                    print(f"Huawei rate limit (407). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                return data # Return error dict if not success and not 407
                
            except Exception as e:
                print(f"Error fetching KPI: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
        return None

    def get_kpi_station_day(self, station_codes, collect_time):
        """
        Retrieves Daily KPI data for stations.
        Endpoint: /thirdData/getKpiStationDay
        collect_time: Long (milliseconds)
        """
        if not self.token:
            if not self.login(): return None
            
        url = f"{self.base_url}/thirdData/getKpiStationDay"
        payload = {
            "stationCodes": ",".join(station_codes) if isinstance(station_codes, list) else station_codes,
            "collectTime": collect_time
        }
        
        # Retry logic internal to Client
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=10)
                data = response.json()
                
                if data.get("success", False):
                    return data.get("data", [])
                
                # Check 407
                if data.get("failCode") == 407:
                    wait = 5 * (attempt + 1)
                    print(f"Huawei rate limit (407). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                return data 
                
            except Exception as e:
                print(f"Error fetching Day KPI: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
        return None

    def get_station_real_kpi(self, station_codes):
        """
        Retrieves Real-Time KPI data for stations.
        Endpoint: /thirdData/getStationRealKpi
        """
        if not self.token:
            if not self.login(): return None
            
        url = f"{self.base_url}/thirdData/getStationRealKpi"
        payload = {
            "stationCodes": ",".join(station_codes) if isinstance(station_codes, list) else station_codes
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=10)
            data = response.json()
            if data.get("success", False):
                return data.get("data", [])
            else:
                print(f"Get RealTime KPI Failed: {data}")
                return None
        except Exception as e:
            print(f"Error fetching RealTime KPI: {e}")
            return None
