import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chat_history.db"

def initialize_database():
    """
    Create the SQLite database and required table
    if it does not already exist.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            ticker TEXT NOT NULL,
            user_query TEXT NOT NULL,

            stock_data TEXT,
            technical_data TEXT,
            news TEXT,

            recommendation TEXT,
            final_report TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_chat(state, final_report):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chat_history(
            ticker,
            user_query,
            stock_data,
            technical_data,
            news,
            recommendation,
            final_report
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            state["ticker"],
            state["user_query"],
            str(state["stock_data"]),
            str(state["technical_data"]),
            state["news"],
            state["recommendation"],
            final_report
        )
    )

    conn.commit()
    conn.close()


def load_chat_history(ticker):
    """
    Load recent chat history.
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM chat_history
        WHERE ticker = ?
        ORDER BY created_at DESC
        LIMIT 5
        """,
        (ticker,)
    )

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]

def clear_chat_history():
    """
    Delete all saved chat history.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM chat_history"
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":

    initialize_database()

    print("SQLite database initialized successfully.")