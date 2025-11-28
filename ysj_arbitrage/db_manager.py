import sqlite3
import datetime
import os
from .config import DB_PATH, PRICE_MARKUP


def _ensure_column(cursor, table, column_definition):
    """添加缺失的列（若存在则忽略）"""
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_definition}")
        print(f"[DB] Added column on {table}: {column_definition}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


def _generate_unique_key(model, config, condition):
    return f"{(model or '').strip()}|{(config or '').strip()}|{(condition or '').strip()}".lower()


def _insert_price_history(cursor, price_id, brand, model, config, condition, original_price,
                          purchase_price, source, batch_id, change_type, raw_price, markup_factor):
    snapshot_time = datetime.datetime.now()
    cursor.execute('''
    INSERT INTO price_history (
        price_id, brand, model, config, condition,
        original_price, purchase_price, change_type,
        source, batch_id, snapshot_time, raw_price, markup_factor
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        price_id, brand, model, config, condition,
        original_price, purchase_price, change_type,
        source, batch_id, snapshot_time, raw_price, markup_factor
    ))

def init_db():
    """初始化数据库，创建表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建价格表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        brand TEXT,
        model TEXT,
        config TEXT,
        condition TEXT,
        purchase_price REAL,
        original_price REAL,
        update_time TIMESTAMP,
        source TEXT DEFAULT 'ocr',
        batch_id TEXT,
        date_key TEXT,
        confidence REAL,
        status TEXT DEFAULT 'active',
        last_sync_at TIMESTAMP,
        unique_key TEXT,
        raw_price REAL,
        markup_factor REAL DEFAULT 1.03
    )
    ''')

    # 旧表补齐缺失列
    _ensure_column(cursor, "prices", "source TEXT DEFAULT 'ocr'")
    _ensure_column(cursor, "prices", "batch_id TEXT")
    _ensure_column(cursor, "prices", "date_key TEXT")
    _ensure_column(cursor, "prices", "confidence REAL")
    _ensure_column(cursor, "prices", "status TEXT DEFAULT 'active'")
    _ensure_column(cursor, "prices", "last_sync_at TIMESTAMP")
    _ensure_column(cursor, "prices", "unique_key TEXT")
    _ensure_column(cursor, "prices", "raw_price REAL")
    _ensure_column(cursor, "prices", "markup_factor REAL DEFAULT 1.03")
    
    # 创建唯一索引，避免重复插入
    cursor.execute('''
    CREATE UNIQUE INDEX IF NOT EXISTS idx_model_config_condition 
    ON prices (model, config, condition)
    ''')

    cursor.execute('''
    CREATE UNIQUE INDEX IF NOT EXISTS idx_prices_unique_key
    ON prices (unique_key)
    ''')

    # 历史表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS price_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        price_id INTEGER,
        brand TEXT,
        model TEXT,
        config TEXT,
        condition TEXT,
        original_price REAL,
        purchase_price REAL,
        change_type TEXT,
        source TEXT,
        batch_id TEXT,
        snapshot_time TIMESTAMP,
        raw_price REAL,
        markup_factor REAL,
        FOREIGN KEY(price_id) REFERENCES prices(id)
    )
    ''')

    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_price_history_price_id
    ON price_history (price_id)
    ''')

    # 填充缺失字段
    cursor.execute('''
    UPDATE prices
    SET unique_key = LOWER(model || '|' || config || '|' || condition)
    WHERE unique_key IS NULL OR unique_key = ''
    ''')
    cursor.execute('''
    UPDATE prices
    SET date_key = COALESCE(date_key, strftime('%Y-%m-%d', COALESCE(update_time, CURRENT_TIMESTAMP)))
    ''')
    cursor.execute('''
    UPDATE prices
    SET status = COALESCE(status, 'active')
    ''')
    cursor.execute('''
    UPDATE prices
    SET markup_factor = COALESCE(markup_factor, 1.03)
    ''')
    cursor.execute('''
    UPDATE prices
    SET raw_price = COALESCE(raw_price, original_price)
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def insert_or_update_price(
    brand,
    model,
    config,
    condition,
    original_price,
    markup_rate=PRICE_MARKUP,
    *,
    source="ocr",
    batch_id=None,
    date_key=None,
    confidence=None,
    status="active",
    last_sync_at=None,
    raw_price=None,
    markup_factor=None,
):
    """插入或更新价格数据，并记录历史"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if raw_price is None:
        raw_price = float(original_price)
    if markup_factor is None:
        markup_factor = markup_rate
    purchase_price = round(float(raw_price) * float(markup_factor), 2)
    original_price = raw_price  # 保持兼容，original_price 仍存 OCR 结果
    update_time = datetime.datetime.now()
    if not date_key:
        date_key = update_time.strftime('%Y-%m-%d')
    if batch_id is None:
        batch_id = f"batch_{update_time.strftime('%Y%m%d')}"
    unique_key = _generate_unique_key(model, config, condition)

    cursor.execute('''
    SELECT id, original_price, purchase_price FROM prices
    WHERE model = ? AND config = ? AND condition = ?
    ''', (model, config, condition))
    existing = cursor.fetchone()

    try:
        cursor.execute('''
        INSERT INTO prices (
            brand, model, config, condition,
            purchase_price, original_price, update_time,
            source, batch_id, date_key, confidence, status,
            last_sync_at, unique_key, raw_price, markup_factor
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model, config, condition)
        DO UPDATE SET
            brand = excluded.brand,
            purchase_price = excluded.purchase_price,
            original_price = excluded.original_price,
            update_time = excluded.update_time,
            source = excluded.source,
            batch_id = excluded.batch_id,
            date_key = excluded.date_key,
            confidence = excluded.confidence,
            status = excluded.status,
            last_sync_at = excluded.last_sync_at,
            unique_key = excluded.unique_key,
            raw_price = excluded.raw_price,
            markup_factor = excluded.markup_factor
        ''', (
            brand, model, config, condition,
            purchase_price, original_price, update_time,
            source, batch_id, date_key, confidence, status,
            last_sync_at, unique_key, raw_price, markup_factor
        ))

        cursor.execute('''
        SELECT id, original_price, purchase_price FROM prices
        WHERE model = ? AND config = ? AND condition = ?
        ''', (model, config, condition))
        row = cursor.fetchone()
        price_id = row[0] if row else None

        price_changed = (
            existing is None or
            round(float(existing[1]), 2) != round(float(original_price), 2) or
            round(float(existing[2]), 2) != round(float(purchase_price), 2)
        )

        if price_id and price_changed:
            change_type = "create" if existing is None else "update"
            _insert_price_history(
                cursor,
                price_id,
                brand, model, config, condition,
                original_price, purchase_price,
                source, batch_id, change_type,
                raw_price, markup_factor
            )

        conn.commit()
        print(f"[DB] Saved: {brand} | {model} | {config} | {condition} | 充新价={original_price} | 收货价={purchase_price}")
        return True
    except Exception as e:
        import traceback
        print(f"[DB] Error inserting data: {e}")
        print(f"[DB] Details: brand={brand}, model={model}, config={config}, condition={condition}, price={original_price}")
        traceback.print_exc()
        return False
    finally:
        conn.close()

def get_target_models(limit=10):
    """获取需要监控的目标机型（这里简单返回所有机型，实际可按利润排序或指定）"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT DISTINCT model FROM prices 
    ORDER BY update_time DESC
    LIMIT ?
    ''', (limit,))
    
    models = [row[0] for row in cursor.fetchall()]
    conn.close()
    return models

def get_all_prices():
    """获取所有价格数据，按更新时间倒序排列"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT brand, model, config, condition, purchase_price, update_time,
           source, batch_id, date_key, confidence, status,
           raw_price, markup_factor
    FROM prices 
    ORDER BY update_time DESC
    ''')
    
    results = cursor.fetchall()
    print(f"DB Query: Found {len(results)} records in {DB_PATH}") # 调试打印
    conn.close()
    return results

def get_price_info(model):
    """根据机型获取该机型的所有配置和成色价格信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT config, condition, purchase_price FROM prices WHERE model = ?
    ''', (model,))
    
    results = cursor.fetchall()
    conn.close()
    return results

def get_price_detail(model, config, condition):
    """根据唯一键获取单条记录的完整信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, brand, model, config, condition, original_price, purchase_price, update_time,
           source, batch_id, date_key, confidence, status, raw_price, markup_factor
    FROM prices WHERE model = ? AND config = ? AND condition = ?
    ''', (model, config, condition))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "id": row[0],
        "brand": row[1],
        "model": row[2],
        "config": row[3],
        "condition": row[4],
        "original_price": row[5],
        "purchase_price": row[6],
        "update_time": row[7],
        "source": row[8],
        "batch_id": row[9],
        "date_key": row[10],
        "confidence": row[11],
        "status": row[12],
        "raw_price": row[13],
        "markup_factor": row[14],
    }

def update_price_record(
    record_id,
    brand,
    model,
    config,
    condition,
    original_price,
    markup_rate=PRICE_MARKUP,
    *,
    source="manual_gui",
    batch_id=None,
    date_key=None,
    confidence=None,
    status="active",
    last_sync_at=None,
    raw_price=None,
    markup_factor=None,
):
    """更新指定记录，并记录历史"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT original_price, purchase_price, raw_price, markup_factor FROM prices WHERE id = ?', (record_id,))
    existing = cursor.fetchone()
    if not existing:
        conn.close()
        return False
    
    if raw_price is None:
        raw_price = float(original_price)
    if markup_factor is None:
        markup_factor = markup_rate if existing is None else (existing[3] or markup_rate)
    purchase_price = round(float(raw_price) * float(markup_factor), 2)
    original_price = raw_price
    update_time = datetime.datetime.now()
    if not date_key:
        date_key = update_time.strftime('%Y-%m-%d')
    if batch_id is None:
        batch_id = f"manual_{update_time.strftime('%Y%m%d')}"
    unique_key = _generate_unique_key(model, config, condition)
    
    cursor.execute('''
    UPDATE prices
    SET brand = ?, model = ?, config = ?, condition = ?,
        original_price = ?, purchase_price = ?, update_time = ?,
        source = ?, batch_id = ?, date_key = ?, confidence = ?, status = ?,
        last_sync_at = ?, unique_key = ?, raw_price = ?, markup_factor = ?
    WHERE id = ?
    ''', (
        brand, model, config, condition,
        original_price, purchase_price, update_time,
        source, batch_id, date_key, confidence, status,
        last_sync_at, unique_key, raw_price, markup_factor, record_id
    ))
    affected = cursor.rowcount
    
    if affected:
        prev_original, prev_purchase, prev_raw, prev_factor = existing
        price_changed = (
            round(float(prev_raw), 2) != round(float(raw_price), 2) or
            round(float(prev_purchase), 2) != round(float(purchase_price), 2) or
            round(float(prev_factor or markup_factor), 3) != round(float(markup_factor), 3)
        )
        if price_changed:
            _insert_price_history(
                cursor,
                record_id,
                brand, model, config, condition,
                original_price, purchase_price,
                source, batch_id, "manual_update",
                raw_price, markup_factor
            )
        conn.commit()
    conn.close()
    return affected > 0

def delete_price_record(record_id):
    """删除指定记录"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM price_history WHERE price_id = ?', (record_id,))
    cursor.execute('DELETE FROM prices WHERE id = ?', (record_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0

