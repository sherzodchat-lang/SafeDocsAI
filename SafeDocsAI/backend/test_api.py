#!/usr/bin/env python3
"""
Test API endpoints
"""
import asyncio
import httpx

BASE_URL = "http://localhost:8001"

async def test_chat():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        # 1. Login
        print("🔑 Logging in...")
        resp = await client.post(
            "/api/v1/auth/login/access-token",
            data={"username": "testuser", "password": "testpass123"}
        )
        if resp.status_code != 200:
            print(f"❌ Login failed: {resp.text}")
            return
        
        token = resp.json()["access_token"]
        print("✅ Logged in")
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Test questions
        questions = [
            "какая ставка налога на прибыль",
            "как уплатить НДС",
            "ставка социального налога",
        ]
        
        for q in questions:
            print(f"\n💬 Question: {q}")
            resp = await client.post(
                "/api/v1/chat/",
                json={"question": q},
                headers=headers
            )
            
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "N/A")
                sources = data.get("sources", [])
                print(f"✅ Answer: {answer[:150]}...")
                print(f"📚 Sources: {len(sources)}")
            else:
                print(f"❌ Error: {resp.status_code} - {resp.text}")

if __name__ == "__main__":
    asyncio.run(test_chat())
