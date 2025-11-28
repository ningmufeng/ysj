import time
import os
import subprocess
from .config import ADB_PATH, XIANYU_PACKAGE, LOGS_DIR
import uiautomator2 as u2


class XianyuBot:
    def __init__(self):
        self.device_id = None
        self.d = None
        self._connect_device()

    def _connect_device(self):
        """优先根据 adb 设备列表连接，再退化为默认连接"""
        serial = None
        try:
            print(f"Checking adb devices using {ADB_PATH}...")
            result = subprocess.run([ADB_PATH, "devices"], capture_output=True, text=True)
            output = result.stdout.strip()
            print(f"ADB Output:\n{output}")
            for line in output.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    serial = parts[0]
                    break
        except Exception as e:
            print(f"Failed to list adb devices: {e}")

        try:
            if serial:
                self.d = u2.connect_usb(serial)
                self.device_id = serial
            else:
                # 回退：尝试连接当前唯一设备
                self.d = u2.connect()
                self.device_id = getattr(self.d, "serial", None)
        except Exception as e:
            print(f"uiautomator2 connect failed: {e}")
            self.d = None

        if self.d:
            print(f"Connected to device: {self.device_id or 'unknown'}")
            # 使用 fast input IME，确保中文输入稳定
            try:
                self.d.set_fastinput_ime(True)
            except Exception:
                pass
        else:
            print("Unable to connect to device via uiautomator2.")

    def start_app(self):
        if not self.d:
            return
        print("Starting Xianyu via uiautomator2...")
        try:
            self.d.app_start(XIANYU_PACKAGE)
            # 等待启动页/广告结束
            time.sleep(12)
        except Exception as e:
            print(f"Failed to start app: {e}")

    def _click_search_box(self):
        """尝试找到搜索框，失败则返回 False"""
        if not self.d:
            return False

        selectors = [
            lambda: self.d(resourceId="com.taobao.idlefish:id/home_page_search_input"),
            lambda: self.d(resourceId="com.taobao.idlefish:id/searchEditText"),
            lambda: self.d(resourceId="com.taobao.idlefish:id/edit_search_keyword"),
            lambda: self.d(resourceId="com.taobao.idlefish:id/rv_search_input"),
            lambda: self.d(descriptionContains="搜索"),
            lambda: self.d(textContains="搜索"),
            lambda: self.d(className="android.widget.EditText", packageName=XIANYU_PACKAGE),
        ]

        for selector in selectors:
            try:
                node = selector()
                if node.exists:
                    try:
                        info = node.info
                        if not info.get("clickable", True):
                            parent = node.xpath("..")
                            if parent.exists:
                                parent.click()
                                return True
                        bounds = info.get("bounds")
                        if bounds:
                            cx = (bounds["left"] + bounds["right"]) / 2
                            cy = (bounds["top"] + bounds["bottom"]) / 2
                            self.d.click(cx, cy)
                            return True
                    except Exception:
                        pass
                    node.click()
                    return True
            except Exception:
                continue

        # 兜底：仍可用按坐标的方式
        try:
            width, height = self.d.window_size()
            x = width * 0.6
            y = height * 0.1
            self.d.click(x, y)
            return True
        except Exception:
            return False

    def _clear_search_input(self):
        """尝试点击“X”按钮清空搜索框"""
        if not self.d:
            return
        clear_ids = [
            "com.taobao.idlefish:id/home_search_clear",
            "com.taobao.idlefish:id/search_clear",
            "com.taobao.idlefish:id/rv_search_clear",
        ]
        for rid in clear_ids:
            btn = self.d(resourceId=rid)
            if btn.exists:
                btn.click()
                time.sleep(0.3)
                return
        # fallback:尝试长按选择全部删除
        edit = self.d(className="android.widget.EditText", packageName=XIANYU_PACKAGE, focused=True)
        if edit.exists:
            try:
                edit.long_click()
                time.sleep(0.2)
                select_all = self.d(text="全选")
                if select_all.exists:
                    select_all.click()
                self.d.press("del")
            except Exception:
                pass

    def search_model(self, keyword):
        """使用 uiautomator2 搜索关键词"""
        if not self.d:
            return False

        if not self.return_to_home():
            print("Failed to navigate back to home, abort search.")
            return False

        print(f"Searching for: {keyword}")
        if not self._click_search_box():
            print("Failed to locate search box.")
            return False

        time.sleep(0.5)
        self._clear_search_input()

        if not self._input_text(keyword):
            print("Failed to input text via all methods.")
            return False

        time.sleep(0.5)
        try:
            self.d.press("enter")
        except Exception:
            pass
        time.sleep(3)

        self.tap_new_tab()
        time.sleep(1.5)
        return True

    def _is_home_screen(self):
        """通过底部导航条检测是否已经回到首页"""
        if not self.d:
            return False
        candidates = [
            lambda: self.d(resourceId="com.taobao.idlefish:id/publish_button"),  # 中央黄按钮
            lambda: self.d(text="卖闲置"),
            lambda: self.d(description="卖闲置"),
            lambda: self.d(text="闲鱼"),
            lambda: self.d(text="消息"),
            lambda: self.d(text="我的"),
        ]
        for getter in candidates:
            try:
                node = getter()
                if node.exists:
                    return True
            except Exception:
                continue
        return False

    def return_to_home(self, max_steps=4):
        """确保回到首页后再继续下一次搜索"""
        if not self.d:
            return False

        for _ in range(max_steps):
            if self._is_home_screen():
                return True
            self.back()
            time.sleep(0.8)

        print("Unable to reach home via back, restarting app...")
        self.start_app()
        return self._is_home_screen()

    def _input_text(self, keyword):
        """多策略输入文本"""
        edit = self.d(className="android.widget.EditText", packageName=XIANYU_PACKAGE, focused=True)
        if edit.exists:
            try:
                edit.set_text(keyword)
                return True
            except Exception as e:
                print(f"set_text failed: {e}")

        try:
            self.d.send_keys(keyword)
            return True
        except Exception as e:
            print(f"send_keys failed: {e}")

        try:
            # fallback to shell input (ASCII only)
            ascii_text = keyword.encode("ascii", errors="ignore").decode("ascii")
            if ascii_text:
                self.d.shell(["input", "text", ascii_text.replace(" ", "%s")])
                return True
        except Exception:
            pass
        return False

    def capture_list_items(self, save_dir=LOGS_DIR):
        """使用 uiautomator2 截图并滑动"""
        if not self.d:
            return []

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        timestamp = int(time.time())
        screenshot_path = os.path.join(save_dir, f"screenshot_{timestamp}.png")
        hierarchy_path = os.path.join(save_dir, f"hierarchy_{timestamp}.xml")

        try:
            self.d.screenshot(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
            time.sleep(0.5)
            if not os.path.exists(screenshot_path) or os.path.getsize(screenshot_path) == 0:
                print("Screenshot file empty, retry later.")
                return []
        except Exception as e:
            print(f"Screenshot failed: {e}")
            return []

        try:
            xml_text = self.d.dump_hierarchy(compressed=False)
            if xml_text:
                with open(hierarchy_path, "w", encoding="utf-8") as f:
                    f.write(xml_text)
            else:
                print("Dump hierarchy returned empty text")
                hierarchy_path = None
        except Exception as e:
            print(f"Dump hierarchy failed: {e}")
            hierarchy_path = None

        # 上滑一屏
        try:
            width, height = self.d.window_size()
            self.d.swipe(width / 2, height * 0.8, width / 2, height * 0.3, 0.3)
            time.sleep(1.5)
        except Exception as e:
            print(f"Swipe failed: {e}")

        return [{
            "screenshot": screenshot_path,
            "hierarchy": hierarchy_path
        }]

    def go_to_detail(self, text_contains):
        if not self.d:
            return False
        try:
            node = self.d(textContains=text_contains)
            if node.exists:
                node.click()
                return True
        except Exception:
            pass
        print("Unable to locate item automatically, please tap manually.")
        return False

    def back(self):
        if not self.d:
            return
        try:
            self.d.press("back")
        except Exception as e:
            print(f"Back key failed: {e}")

    def tap_new_tab(self):
        """在搜索结果页点击“新发”"""
        if not self.d:
            return False
        selectors = [
            lambda: self.d(text="新发"),
            lambda: self.d(description="新发"),
            lambda: self.d(textContains="新发"),
            lambda: self.d(resourceId="com.taobao.idlefish:id/search_new_box"),
        ]
        for getter in selectors:
            try:
                node = getter()
                if node.exists:
                    node.click()
                    return True
            except Exception:
                continue
        print("New tab (新发) not found on current screen.")
        return False


if __name__ == "__main__":
    bot = XianyuBot()
    bot.start_app()
    bot.search_model("Reno 14 pro 演示机")
    bot.capture_list_items()
