import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import csv
import os
from datetime import datetime
from .db_manager import (
    init_db,
    get_target_models,
    get_all_prices,
    insert_or_update_price,
    get_price_detail,
    update_price_record,
    delete_price_record,
)
from .ocr_parser import process_input_images
from .xianyu_bot import XianyuBot
from .logic_manager import ArbitrageLogic
from .config import RESULTS_DIR, PRICE_MARKUP

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("演示机套利助手")
        self.root.geometry("1000x600")
        
        self.bot = XianyuBot()
        self.logic = ArbitrageLogic()
        self.stop_event = threading.Event() # 用于控制任务停止
        
        # 初始化数据库
        init_db()
        
        self.setup_ui()
        self.monitoring = False
        self.csv_file = os.path.join(RESULTS_DIR, f"results_{datetime.now().strftime('%Y%m%d')}.csv")
        self._ensure_csv_header()
        self._db_menu = None

    def setup_ui(self):
        # 顶部控制栏
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        ttk.Button(control_frame, text="1. 导入报价单", command=self.import_quotes).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="2. 启动监控", command=self.toggle_monitor).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="停止所有任务", command=self.stop_tasks).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="查看报价库", command=self.show_database_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="连接手机", command=self.connect_device).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="就绪")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # 主内容区
        content_frame = ttk.Frame(self.root, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：监控结果列表
        left_frame = ttk.LabelFrame(content_frame, text="实时监控结果", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        columns = ("title", "price", "ref_price", "profit")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings")
        self.tree.heading("title", text="标题")
        self.tree.heading("price", text="挂价")
        self.tree.heading("ref_price", text="收货价")
        self.tree.heading("profit", text="利润")
        
        self.tree.column("title", width=300)
        self.tree.column("price", width=80)
        self.tree.column("ref_price", width=80)
        self.tree.column("profit", width=80)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_item_click)
        
        # 右侧：日志
        right_frame = ttk.LabelFrame(content_frame, text="运行日志", padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        self.log_text = tk.Text(right_frame, width=40, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)

    def connect_device(self):
        if self.bot.device_id:
            self.log(f"已连接: {self.bot.device_id}")
        else:
            self.bot = XianyuBot()
            if self.bot.device_id:
                self.log(f"连接成功: {self.bot.device_id}")
            else:
                self.log("连接失败，请检查 ADB")

    def _notify_status(self, log_message=None, status_text=None):
        """线程安全地更新日志与状态标签"""
        def updater():
            if log_message:
                self.log(log_message)
            if status_text:
                self.status_label.config(text=status_text)
        self.root.after(0, updater)

    def import_quotes(self):
        self.stop_event.clear()
        threading.Thread(target=self._import_task).start()

    def stop_tasks(self):
        self.stop_event.set()
        self.monitoring = False
        self.log("正在停止任务...")
        self.status_label.config(text="正在停止...")

    def _import_task(self):
        self._notify_status("开始导入报价单...", "正在导入...")
        process_input_images(stop_event=self.stop_event)
        if not self.stop_event.is_set():
            self._notify_status("导入完成", "导入完成")
            # 如果数据库窗口已打开，自动刷新
            if hasattr(self, '_db_window') and self._db_window.winfo_exists():
                self.root.after(0, self._refresh_database_window)
                self.root.after(0, lambda: messagebox.showinfo("提示", "报价单导入完成！\n数据库窗口已自动刷新。"))
            else:
                self.root.after(0, lambda: messagebox.showinfo("提示", "报价单导入完成！\n请点击'查看报价库'查看导入的数据。"))
        else:
            self._notify_status("导入已停止", "已停止")

    def toggle_monitor(self):
        if self.monitoring:
            self.stop_tasks()
            self.status_label.config(text="监控已停止")
        else:
            self.monitoring = True
            self.stop_event.clear()
            self.status_label.config(text="监控中...")
            threading.Thread(target=self._monitor_loop).start()

    def _monitor_loop(self):
        self._notify_status("监控线程启动", None)
        self.bot.start_app()
        
        # 获取要监控的机型
        target_models = get_target_models()
        if not target_models:
            self._notify_status("数据库无目标机型，请先导入报价单", "监控已停止")
            self.monitoring = False
            return
        try:
            while self.monitoring:
                if self.stop_event.is_set():
                    break
                
                for model in target_models:
                    if not self.monitoring or self.stop_event.is_set():
                        break
                    
                    self._notify_status(f"正在搜索: {model}", None)
                    if self.bot.search_model(f"{model} 演示机"):
                        screenshots = self.bot.capture_list_items()
                        
                        for shot in screenshots:
                            if self.stop_event.is_set():
                                break
                            screenshot_path = shot.get("screenshot") if isinstance(shot, dict) else shot
                            hierarchy_path = shot.get("hierarchy") if isinstance(shot, dict) else None
                            items = self.logic.analyze_screenshot(screenshot_path, model, hierarchy_path)
                            for item in items:
                                self.add_result(item)
                                if item['level'] in ['high', 'medium']:
                                    self._notify_status(f"发现利润单: {item['title']} (利润 {item['profit']})", None)
                        
                        if not self.stop_event.is_set():
                            self.bot.back() # 退出搜索页
                    
                    time.sleep(5) # 间隔
        finally:
            self.monitoring = False
            msg = "监控已停止" if self.stop_event.is_set() else "监控结束"
            self._notify_status(msg, msg)

    def _ensure_csv_header(self):
        """确保 CSV 文件存在且有表头"""
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            if not os.path.exists(self.csv_file):
                with open(self.csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(['时间', '标题', '价格', '参考价', '利润', '等级'])
        except Exception as e:
            print(f"初始化 CSV 文件失败: {e}")
    
    def add_result(self, item):
        tags = (item['level'],)
        self.tree.insert("", 0, values=(item['title'], item['price'], item['ref_price'], item['profit']), tags=tags)
        # 设置颜色
        self.tree.tag_configure('high', background='#90EE90') # 浅绿
        self.tree.tag_configure('medium', background='#FFFFE0') # 浅黄
        
        # 保存到 CSV
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    item['title'],
                    item['price'],
                    item['ref_price'],
                    item['profit'],
                    item['level']
                ])
        except Exception as e:
            print(f"保存 CSV 失败: {e}")

    def show_database_window(self):
        """显示数据库内容窗口"""
        # 检查是否已有窗口打开
        if hasattr(self, '_db_window') and self._db_window.winfo_exists():
            # 如果窗口已存在，刷新数据
            self._refresh_database_window()
            self._db_window.lift()  # 将窗口置于最前
            return
        
        db_window = tk.Toplevel(self.root)
        db_window.title("演示机报价库")
        db_window.geometry("800x500")
        self._db_window = db_window  # 保存窗口引用
        
        # 创建顶部工具栏
        toolbar = ttk.Frame(db_window, padding="5")
        toolbar.pack(fill=tk.X)
        
        refresh_btn = ttk.Button(toolbar, text="刷新", command=self._refresh_database_window)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建表格
        columns = (
            "brand",
            "model",
            "config",
            "condition",
            "raw_price",
            "markup_factor",
            "price",
            "time",
            "source",
            "batch_id",
            "date_key",
            "confidence",
            "status",
        )
        tree = ttk.Treeview(db_window, columns=columns, show="headings", selectmode="extended")
        self._db_tree = tree  # 保存表格引用
        
        tree.heading("brand", text="品牌")
        tree.heading("model", text="型号")
        tree.heading("config", text="配置")
        tree.heading("condition", text="成色")
        tree.heading("raw_price", text="原始价格")
        tree.heading("markup_factor", text="计算系数")
        tree.heading("price", text="收货价")
        tree.heading("time", text="更新时间")
        tree.heading("source", text="来源")
        tree.heading("batch_id", text="批次")
        tree.heading("date_key", text="日期")
        tree.heading("confidence", text="置信度")
        tree.heading("status", text="状态")
        
        tree.column("brand", width=80)
        tree.column("model", width=200)
        tree.column("config", width=100)
        tree.column("condition", width=80)
        tree.column("raw_price", width=90)
        tree.column("markup_factor", width=90)
        tree.column("price", width=90)
        tree.column("time", width=140)
        tree.column("source", width=80)
        tree.column("batch_id", width=140)
        tree.column("date_key", width=90)
        tree.column("confidence", width=80)
        tree.column("status", width=80)
        
        # 添加滚动条
        v_scrollbar = ttk.Scrollbar(db_window, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(db_window, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscroll=v_scrollbar.set, xscroll=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True)
        tree.bind("<Button-3>", self._on_db_right_click)
        
        # 右键菜单
        self._db_menu = tk.Menu(db_window, tearoff=0)
        self._db_menu.add_command(label="增加", command=lambda: self._open_record_dialog(mode="add"))
        self._db_menu.add_command(label="修改", command=lambda: self._open_record_dialog(mode="edit"))
        self._db_menu.add_command(label="删除", command=self._delete_selected_record)
        
        # 加载数据
        self._refresh_database_window()
        
        # 窗口关闭时清理引用
        def on_close():
            if hasattr(self, '_db_window'):
                delattr(self, '_db_window')
            if hasattr(self, '_db_tree'):
                delattr(self, '_db_tree')
            if hasattr(self, '_db_menu'):
                delattr(self, '_db_menu')
            db_window.destroy()
        db_window.protocol("WM_DELETE_WINDOW", on_close)
    
    def _refresh_database_window(self):
        """刷新数据库窗口的数据"""
        if not hasattr(self, '_db_tree'):
            print("[GUI] 数据库窗口未打开，无法刷新")
            return
        
        try:
            # 检查窗口是否还存在
            if not self._db_tree.winfo_exists():
                print("[GUI] 数据库窗口已关闭，无法刷新")
                return
        except tk.TclError:
            print("[GUI] 数据库窗口已关闭，无法刷新")
            return
        
        # 清空现有数据
        for item in self._db_tree.get_children():
            self._db_tree.delete(item)
        
        # 重新加载数据
        prices = get_all_prices()
        print(f"[GUI] 从数据库加载了 {len(prices)} 条记录")
        
        for p in prices:
            self._db_tree.insert("", tk.END, values=p)
        
        print(f"[GUI] 数据库窗口已刷新，共显示 {len(prices)} 条记录")

    def _on_db_right_click(self, event):
        """数据库表格右键菜单"""
        if not hasattr(self, '_db_tree') or not hasattr(self, '_db_menu'):
            return
        
        row_id = self._db_tree.identify_row(event.y)
        if row_id:
            if row_id not in self._db_tree.selection():
                self._db_tree.selection_set(row_id)
            self._db_tree.focus(row_id)
        else:
            self._db_tree.selection_remove(self._db_tree.selection())
        try:
            self._db_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._db_menu.grab_release()

    def _open_record_dialog(self, mode="add"):
        """打开新增/编辑记录对话框"""
        parent = self._db_window if hasattr(self, '_db_window') else self.root
        record = None
        
        if mode != "add":
            selection = self._db_tree.selection() if hasattr(self, '_db_tree') else ()
            if not selection:
                messagebox.showwarning("提示", "请先选择一条记录")
                return
            values = self._db_tree.item(selection[0])['values']
            model, config, condition = values[1], values[2], values[3]
            record = get_price_detail(model, config, condition)
            if not record:
                messagebox.showerror("错误", "未找到对应的数据库记录，可能已被删除。")
                return
        
        dialog = tk.Toplevel(parent)
        dialog.title("新增记录" if mode == "add" else "修改记录")
        dialog.geometry("360x260")
        dialog.resizable(False, False)
        dialog.transient(parent)
        dialog.grab_set()
        
        brand_var = tk.StringVar(value=record['brand'] if record else "")
        model_var = tk.StringVar(value=record['model'] if record else "")
        config_var = tk.StringVar(value=record['config'] if record else "")
        condition_var = tk.StringVar(value=record['condition'] if record else "充新")
        original_var = tk.StringVar(value=str(record['raw_price']) if record and record.get('raw_price') else "")
        markup_var = tk.StringVar(
            value=str(record['markup_factor']) if record and record.get('markup_factor') else f"{PRICE_MARKUP:.2f}"
        )
        purchase_var = tk.StringVar(value="--")
        
        def update_purchase(*args):
            try:
                val = float(original_var.get())
                factor = float(markup_var.get())
                purchase_var.set(f"{round(val * factor, 2):.2f}")
            except ValueError:
                purchase_var.set("--")
        original_var.trace_add("write", update_purchase)
        markup_var.trace_add("write", update_purchase)
        update_purchase()
        
        form_items = [
            ("品牌", brand_var),
            ("型号", model_var),
            ("配置", config_var),
            ("成色", condition_var),
            ("原始价格", original_var),
            ("计算系数", markup_var),
        ]
        
        for idx, (label, var) in enumerate(form_items):
            ttk.Label(dialog, text=label).grid(row=idx, column=0, sticky="e", padx=10, pady=5)
            ttk.Entry(dialog, textvariable=var, width=25).grid(row=idx, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="收货价(自动)").grid(row=len(form_items), column=0, sticky="e", padx=10, pady=5)
        ttk.Label(dialog, textvariable=purchase_var, foreground="#007700").grid(row=len(form_items), column=1, sticky="w", padx=10, pady=5)
        
        def on_save():
            brand = brand_var.get().strip()
            model = model_var.get().strip()
            config = config_var.get().strip()
            condition = condition_var.get().strip()
            original_text = original_var.get().strip()
            markup_text = markup_var.get().strip()
            
            if not all([brand, model, config, condition, original_text]):
                messagebox.showwarning("提示", "请完整填写所有字段")
                return
            try:
                original_price = float(original_text)
                if original_price <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("提示", "原始价格需要为大于0的数字")
                return
            try:
                markup_factor = float(markup_text)
                if markup_factor <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("提示", "计算系数需要为大于0的数字")
                return
            
            batch_id = f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            date_key = datetime.now().strftime('%Y-%m-%d')
            confidence = 1.0
            status_value = (record.get("status") if record and record.get("status") else "manual")
            
            if mode == "add":
                success = insert_or_update_price(
                    brand,
                    model,
                    config,
                    condition,
                    original_price,
                    source="manual_gui",
                    batch_id=batch_id,
                    date_key=date_key,
                    confidence=confidence,
                    status=status_value,
                    raw_price=original_price,
                    markup_factor=markup_factor,
                )
            else:
                success = update_price_record(
                    record["id"],
                    brand,
                    model,
                    config,
                    condition,
                    original_price,
                    source="manual_gui",
                    batch_id=batch_id,
                    date_key=date_key,
                    confidence=confidence,
                    status=status_value,
                    raw_price=original_price,
                    markup_factor=markup_factor,
                )
            
            if success:
                messagebox.showinfo("成功", "记录已保存")
                dialog.destroy()
                self._refresh_database_window()
            else:
                messagebox.showerror("错误", "保存失败，请检查日志")
        
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=len(form_items)+1, column=0, columnspan=2, pady=15)
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="确定", command=on_save).pack(side=tk.RIGHT, padx=5)
        dialog.bind("<Return>", lambda _e: on_save())
        dialog.bind("<Escape>", lambda _e: dialog.destroy())

    def _delete_selected_record(self):
        """删除选中记录"""
        if not hasattr(self, '_db_tree'):
            return
        selection = self._db_tree.selection()
        if not selection:
            messagebox.showwarning("提示", "请先选择一条记录")
            return
        
        records = []
        missing = []
        for item_id in selection:
            values = self._db_tree.item(item_id)['values']
            brand, model, config, condition = values[0], values[1], values[2], values[3]
            record = get_price_detail(model, config, condition)
            if record:
                records.append((record["id"], brand, model, config, condition))
            else:
                missing.append(f"{brand} {model} {config} ({condition})")
        
        if not records:
            messagebox.showerror("错误", "未找到所选记录对应的数据库条目，可能已被删除。")
            return
        
        preview = "\n".join(f"{b} {m} {c} ({d})" for _, b, m, c, d in records[:5])
        if len(records) > 5:
            preview += f"\n... 共 {len(records)} 条"
        if not messagebox.askyesno("确认删除", f"确定删除以下 {len(records)} 条记录？\n{preview}"):
            return
        
        success = 0
        for record_id, *_ in records:
            if delete_price_record(record_id):
                success += 1
        
        failed = len(records) - success
        msg_parts = []
        if success:
            msg_parts.append(f"{success} 条记录已删除")
        if failed:
            msg_parts.append(f"{failed} 条删除失败，请查看日志")
        if missing:
            msg_parts.append(f"{len(missing)} 条在数据库中未找到")
        if msg_parts:
            messagebox.showinfo("删除完成", "\n".join(msg_parts))
        
        self._refresh_database_window()

    def on_item_click(self, event):
        item_id = self.tree.selection()[0]
        item = self.tree.item(item_id)
        title = item['values'][0]
        
        if messagebox.askyesno("操作", f"要在手机上查看 '{title}' 吗？"):
            # 尝试点击
            # 注意：这里只是简单的尝试点击文字，列表页可能因为滑动位置变化而点不到
            self.bot.go_to_detail(title[:10]) # 截取前几个字尝试匹配

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

