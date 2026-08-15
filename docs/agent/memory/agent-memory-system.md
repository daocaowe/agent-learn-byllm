# Agent 记忆系统：短期记忆、工作记忆、长期记忆与外部记忆

> 本文整理自 Agent 开发学习过程中关于 Memory System 的讨论，并以“自助台球客服 Agent”为贯穿案例。

## 1. 核心概念

Agent 的记忆系统不要简单理解成“聊天历史”。可以拆成四类：

| 类型 | 解决的问题 | 典型内容 | 生命周期 |
|---|---|---|---|
| 短期记忆 | 当前对话聊了什么 | messages / conversation history | 当前 Session / Thread |
| 工作记忆 | 当前任务做到哪一步 | State、任务参数、工具结果、中间状态 | 当前任务 |
| 长期记忆 | 跨 Session 还记得用户什么 | 用户偏好、长期事实、目标 | 持久化 |
| 外部记忆 | Agent 可以从外部世界查到什么 | RAG、知识库、业务文档 | 持久化 |

一个重要判断：

> Memory 的分类由信息的语义和生命周期决定，而不是由 Redis、PostgreSQL、Qdrant 等存储介质决定。

例如 Redis 可以存短期 State，PostgreSQL 也可以存短期 State；长期 Memory 可以使用 PostgreSQL，也可以结合向量检索。

---

## 2. 自助台球客服场景

假设有一家无人台球厅，用户可以通过 App / 微信与 Agent 对话：

```text
用户：帮我订今晚8点的2号桌。
Agent：好的，请问打几个小时？
用户：2小时。
```

Agent 可以调用：

```text
check_table_availability()  查询桌位
query_membership()          查询会员
query_price()               查询价格
book_table()                创建预约
cancel_booking()            取消预约
```

同时系统可能包含：

```text
短期记忆       LangGraph Checkpointer
工作记忆       LangGraph State
长期记忆       PostgreSQL / LangGraph Store
外部记忆       Qdrant / Vector DB / Knowledge Base
```

---

# 3. 短期记忆：Conversation History

短期记忆保存当前会话中的消息：

```text
User：我要订2号桌。
Assistant：好的，请问什么时候？
User：今晚8点。
Assistant：好的，请问打几个小时？
User：2小时。
```

对应的概念结构：

```python
messages = [
    HumanMessage("我要订2号桌"),
    AIMessage("好的，请问什么时候？"),
    HumanMessage("今晚8点"),
    AIMessage("好的，请问打几个小时？"),
    HumanMessage("2小时"),
]
```

如果没有历史消息，最后一句“2小时”无法知道指的是什么。

LangGraph 中通常通过 `messages` + `checkpointer` + `thread_id` 实现 Thread / Session 级短期记忆。

```python
config = {
    "configurable": {
        "thread_id": "user_001_session_001"
    }
}
```

同一个 `thread_id` 的后续请求可以恢复之前的 State。

---

# 4. 工作记忆：Working Memory

复杂 Agent 不只是聊天，还要执行任务。例如：

```text
用户：帮我订今晚8点的2号桌，打两个小时。

Step 1：解析需求
Step 2：查询桌位
Step 3：查询会员
Step 4：查询价格
Step 5：计算折扣价格
Step 6：创建订单
Step 7：返回订单号
```

Agent 必须记住当前任务状态：

```python
working_memory = {
    "task": "booking",
    "table_id": 2,
    "booking_time": "20:00",
    "duration": 2,
    "available": True,
    "membership_level": "VIP",
    "discount": 0.8,
    "price": 80,
    "booking_id": None,
    "current_step": "create_booking",
}
```

在 LangGraph 中，工作记忆通常落在 `AgentState` 上。

> Conversation History 更像“原始记忆”；Working Memory 更像“整理后的当前任务状态”。

---

# 5. 工具调用与工作记忆

Agent 的工具调用结果也是工作记忆的重要组成部分。

例如：

```python
check_table_availability(
    table_id=2,
    booking_time="20:00",
    duration=2,
)
```

工具返回：

```json
{
  "available": true,
  "table_id": 2
}
```

应该形成：

```text
Tool
 ↓
Tool Result
 ↓
Working Memory / State
 ↓
下一次 Agent 推理
```

再调用：

```python
query_membership(user_id="123")
```

返回：

```json
{
  "level": "VIP",
  "discount": 0.8
}
```

工作状态继续更新：

```python
state["membership_level"] = "VIP"
state["discount"] = 0.8
```

因此 Agent Loop 可以抽象成：

```text
User
 ↓
LLM / Agent
 ↓
决定 Tool
 ↓
Tool
 ↓
Tool Result
 ↓
Working Memory / State
 ↓
LLM / Agent
 ↓
决定下一步 Tool
 ↓
...
 ↓
Final Answer
```

这也是 Agent 与普通 Chatbot 的重要区别。

---

# 6. AgentState 示例

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 短期记忆：当前对话消息
    messages: Annotated[list, add_messages]

    # 用户与项目 Scope
    user_id: str
    project_id: str

    # 当前任务
    task: str | None

    # 预约参数
    table_id: int | None
    booking_time: str | None
    duration: int | None

    # 工具结果
    available: bool | None

    # 会员信息
    membership_level: str | None
    discount: float | None

    # 价格与订单
    price: float | None
    booking_id: str | None

    # Agent 当前阶段
    current_step: str | None
```

其中：

```text
messages
```

主要属于 Conversation History，而：

```text
task
table_id
booking_time
duration
available
membership_level
discount
price
booking_id
current_step
```

主要属于 Working Memory。

---

# 7. 长期记忆：Long-term Memory

如果用户说：

```text
我是张三，我经常周末晚上来打台球。
```

这条信息可能在未来 Session 中仍然有价值，因此可以提取成长期记忆：

```json
{
  "user_id": "123",
  "memory": "用户经常周末晚上打台球",
  "type": "preference",
  "scope": "global"
}
```

长期记忆不是“把所有聊天记录永久保存”。

例如：

```text
“我今天8点来打台球”
```

通常没有必要保存。

而：

```text
“我一般周末晚上来打球。”
“我的项目后端都使用 FastAPI。”
```

更适合成为长期 Memory。

典型生命周期：

```text
Conversation
 ↓
Memory Extraction
 ↓
判断是否具有长期价值
 ↓
结构化 Memory
 ↓
去重 / 更新
 ↓
Persistent Store
```

---

# 8. Global Scope 与 Project Scope

Scope 的核心问题：

> 这条 Memory 到底对谁、在哪个范围内有效？

## Global Scope

Global Memory 对用户所有项目有效：

```text
User 123
│
└── Global Memory
    ├── 喜欢中文
    ├── 喜欢 Python
    └── 通常打2小时
```

例如：

```json
{
  "user_id": "123",
  "project_id": null,
  "content": "用户喜欢中文交流",
  "scope": "global"
}
```

## Project Scope

Project Memory 只在特定项目中有效：

```text
User 123
│
├── Global
│   └── 喜欢中文
│
├── Project: billiard
│   └── 喜欢2号球桌
│
└── Project: english-learning
    └── 喜欢 TTS 练习
```

例如：

```json
{
  "user_id": "123",
  "project_id": "billiard",
  "content": "用户在台球项目中喜欢2号桌",
  "scope": "project"
}
```

这样可以避免不同项目之间发生 Memory Leakage / Memory Contamination。

---

# 9. LangGraph Store 的 Scope 示例

可以用 namespace 表达 Scope：

```python
from langgraph.store.memory import InMemoryStore

memory_store = InMemoryStore()
```

Global：

```python
namespace = ("user", user_id, "global")
```

Project：

```python
namespace = (
    "user",
    user_id,
    "project",
    project_id,
)
```

保存：

```python
memory_store.put(
    namespace,
    "preference_001",
    {
        "type": "preference",
        "content": "用户通常打2小时",
    },
)
```

读取：

```python
memories = memory_store.search(namespace)
```

生产环境可以把这个 Store 替换为持久化存储。

---

# 10. External Memory：RAG / Knowledge Base

台球厅还有业务知识：

```text
收费标准.pdf
会员制度.pdf
退款规则.pdf
营业时间.pdf
台球厅规则.pdf
```

用户问：

```text
VIP晚上8点打两个小时多少钱？
```

此时应该查 Knowledge Base：

```text
User Query
 ↓
Embedding / BM25
 ↓
Vector DB / Search
 ↓
相关业务文档
 ↓
LLM
 ↓
Answer
```

这属于 External Memory。

核心区别：

| Long-term Memory | RAG / External Memory |
|---|---|
| 记住用户 | 记住外部世界知识 |
| 用户偏好 | 公司文档 |
| 用户长期事实 | 产品说明 |
| 用户目标 | 规则、价格、API 文档 |
| 跨 Session | 通常长期存在 |

一句话：

> **Memory 记住“你”，RAG 记住“世界”。**

---

# 11. Memory Read：什么时候读取

不要每次把所有 Memory 全部塞给 LLM：

```text
1000 条 Memory
 ↓
全部放 Prompt
 ↓
LLM
```

这样会带来：

- Token 浪费
- 延迟增加
- 无关信息噪声
- 错误关联
- Context Window 压力

更合理：

```text
Current Query
 ↓
Memory Retrieval
 ↓
相关 Memory
 ↓
Ranking / Top-K
 ↓
LLM Context
```

例如用户问“帮我订个位置”，只检索与预约相关的用户偏好，而不是把所有历史记忆都发送给模型。

因此：

> Memory System 本质上也包含一个 Retrieval 系统。

---

# 12. Memory Write：什么时候写

典型流程：

```text
User Message
 ↓
Memory Extractor
 ↓
重要性判断
 ↓
Scope 判断
 ↓
去重 / 更新
 ↓
Memory Store
```

例如：

```text
“以后我的项目都使用 FastAPI。”
```

可能保存为：

```json
{
  "content": "用户偏好 FastAPI",
  "type": "preference",
  "scope": "global",
  "importance": 0.9
}
```

而：

```text
“这个 PDF 项目使用 Qdrant。”
```

更适合：

```json
{
  "content": "PDF RAG 项目使用 Qdrant",
  "type": "project_fact",
  "scope": "project:pdf-rag"
}
```

---

# 13. 自助台球客服的完整 Memory 架构

```text
                         User
                           │
                           ↓
                     Current Query
                           │
                 ┌─────────┼─────────┐
                 ↓         ↓         ↓
           Short-term   Long-term   External
             Memory      Memory      Memory
                 │         │           │
                 │     ┌───┴───┐       │
                 │     ↓       ↓       │
                 │  Global  Project    │
                 │  Scope    Scope     │
                 │     │       │       │
                 └─────┴───────┴───────┘
                           │
                           ↓
                       Agent / LLM
                           │
                           ↓
                     Working Memory
                         / State
                           │
                     ┌─────┴─────┐
                     ↓           ↓
                   Tool        Tool
                     ↓           ↓
                 查询桌位      查会员
                     │           │
                     └─────┬─────┘
                           ↓
                      Tool Results
                           ↓
                    Working Memory
                           ↓
                     Agent继续执行
                           │
                           ↓
                      Final Answer
                           │
                           ↓
                  Memory Write / Update
```

---

# 14. LangGraph Agent Demo

## Tools

```python
from langchain_core.tools import tool

TABLES = {
    1: [],
    2: ["20:00"],
    3: [],
    4: [],
}


@tool
def check_table_availability(
    table_id: int,
    booking_time: str,
    duration: int,
) -> dict:
    """查询指定时间段球桌是否可用。"""
    booked_times = TABLES.get(table_id, [])

    if booking_time in booked_times:
        return {
            "available": False,
            "message": f"{table_id}号桌 {booking_time} 已被预约",
        }

    return {
        "available": True,
        "table_id": table_id,
        "booking_time": booking_time,
        "duration": duration,
    }


@tool
def query_membership(user_id: str) -> dict:
    """查询用户会员信息。"""
    return {
        "user_id": user_id,
        "level": "VIP",
        "discount": 0.8,
    }


@tool
def query_price(duration: int) -> dict:
    """查询球桌价格。"""
    return {
        "hourly_price": 50,
        "duration": duration,
        "original_price": duration * 50,
    }


@tool
def book_table(
    user_id: str,
    table_id: int,
    booking_time: str,
    duration: int,
) -> dict:
    """创建球桌预约订单。"""
    booking_id = (
        f"B{user_id}{table_id}"
        f"{booking_time.replace(':', '')}"
    )

    TABLES[table_id].append(booking_time)

    return {
        "success": True,
        "booking_id": booking_id,
        "table_id": table_id,
        "booking_time": booking_time,
        "duration": duration,
    }
```

## State

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    project_id: str

    task: str | None

    table_id: int | None
    booking_time: str | None
    duration: int | None

    available: bool | None

    membership_level: str | None
    discount: float | None

    price: float | None
    booking_id: str | None

    current_step: str | None
```

## Memory Store

```python
from langgraph.store.memory import InMemoryStore

memory_store = InMemoryStore()


def save_global_memory(user_id: str, key: str, value: dict):
    namespace = ("user", user_id, "global")
    memory_store.put(namespace, key, value)


def save_project_memory(
    user_id: str,
    project_id: str,
    key: str,
    value: dict,
):
    namespace = (
        "user",
        user_id,
        "project",
        project_id,
    )
    memory_store.put(namespace, key, value)


def get_global_memories(user_id: str):
    namespace = ("user", user_id, "global")
    return memory_store.search(namespace)


def get_project_memories(user_id: str, project_id: str):
    namespace = (
        "user",
        user_id,
        "project",
        project_id,
    )
    return memory_store.search(namespace)
```

## Agent Graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver


tool_node = ToolNode(tools)


def should_continue(state):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer,
)
```

调用：

```python
config = {
    "configurable": {
        "thread_id": "user_001_session_001"
    }
}

result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我订今晚8点的位置",
            }
        ],
        "user_id": "user_001",
        "project_id": "billiard",
        "task": "booking",
        "table_id": None,
        "booking_time": None,
        "duration": None,
        "available": None,
        "membership_level": None,
        "discount": None,
        "price": None,
        "booking_id": None,
        "current_step": "start",
    },
    config,
)
```

---

# 15. 一个完整预约过程中的状态变化

用户：

```text
帮我订今晚8点的2号桌，打两个小时。
```

初始 Working Memory：

```text
task = booking
table_id = 2
booking_time = 20:00
duration = 2
available = None
membership_level = None
price = None
booking_id = None
```

调用：

```text
check_table_availability()
```

结果：

```text
available = True
```

调用：

```text
query_membership()
```

结果：

```text
membership_level = VIP
discount = 0.8
```

调用：

```text
query_price()
```

结果：

```text
original_price = 100
```

计算：

```text
100 × 0.8 = 80
```

调用：

```text
book_table()
```

结果：

```text
booking_id = Buser00122000
```

最终 Working Memory：

```text
task = booking
table_id = 2
booking_time = 20:00
duration = 2
available = True
membership_level = VIP
discount = 0.8
price = 80
booking_id = Buser00122000
current_step = completed
```

---

# 16. 四种 Memory 最终对照

```text
Agent Memory
│
├── Short-term Memory
│   └── messages
│       “我要订2号桌”
│       “今晚8点”
│       “2小时”
│
├── Working Memory
│   └── AgentState
│       table_id = 2
│       booking_time = 20:00
│       duration = 2
│       available = True
│       price = 80
│       booking_id = xxx
│
├── Long-term Memory
│   ├── Global
│   │   ├── 用户喜欢中文
│   │   └── 用户通常打2小时
│   │
│   └── Project: billiard
│       └── 用户喜欢2号桌
│
└── External Memory
    └── RAG / Knowledge Base
        ├── 收费标准
        ├── 会员制度
        ├── 退款规则
        └── 台球厅规则
```

---

# 17. Memory Lifecycle

真正工程化的 Memory 系统，重点不只是“怎么存”，而是整个生命周期：

```text
什么时候读取？
      ↓
读取什么？
      ↓
如何检索 / 排序？
      ↓
什么时候写入？
      ↓
写入什么？
      ↓
属于 Global 还是 Project？
      ↓
保存多久？
      ↓
什么时候更新？
      ↓
什么时候删除？
```

因此成熟 Agent 的架构更接近：

```text
Agent
│
├── Reasoning
├── Tools
├── Short-term Memory
│   └── Conversation History
├── Working Memory
│   └── State / Task State / Tool Results
├── Long-term Memory
│   ├── Global
│   └── Project
└── External Memory
    └── RAG / Knowledge Base
```

---

# 18. 工程化升级路线

当前 Demo 使用内存组件，是为了学习概念。进一步生产化可以升级为：

```text
FastAPI
   +
LangGraph
   +
PostgreSQL
   +
Redis
   +
Qdrant
```

建议职责：

```text
FastAPI
└── API 层

LangGraph
└── Agent Orchestration

Redis
└── 可选：高频 Session / Cache / 临时状态

PostgreSQL
└── 用户、订单、长期 Memory、业务数据

Qdrant
└── RAG / Memory Semantic Retrieval
```

注意：这不是强制绑定。存储介质应根据一致性、TTL、查询方式、规模和运维要求选择。

---

# 19. 学习重点

学习 Agent Memory 时，建议重点掌握以下 8 个概念：

1. Conversation History
2. Working Memory / State
3. Long-term Memory
4. External Memory / RAG
5. Memory Write
6. Memory Read / Retrieval
7. Scope
8. Memory Lifecycle

尤其要记住三个区别：

> **Memory ≠ Chat History**

> **Memory ≠ RAG**

> **Memory = 信息的持久化 + 生命周期管理 + 按需检索 + Scope 管理**

---

# 20. 最终心智模型

可以把 Agent Memory 类比成人脑：

| Agent | 类比 |
|---|---|
| Conversation History | 刚才聊了什么 |
| Working Memory | 现在脑子里正在处理什么 |
| Long-term Memory | 我认识这个人什么 |
| Project Memory | 我对这个项目知道什么 |
| RAG | 我可以查什么资料 |
| Memory Retrieval | 回忆 |
| Memory Write | 记住 |
| Scope | 这件事适用于谁 |
| Ranking | 回忆最相关的信息 |

最终形成：

```text
用户请求
   ↓
读取短期记忆
   ↓
读取相关长期记忆
   ↓
读取相关 RAG
   ↓
Agent 推理
   ↓
Working Memory / State
   ↓
调用 Tool
   ↓
Tool Result
   ↓
更新 Working Memory
   ↓
继续 Agent Loop
   ↓
最终回答
   ↓
判断是否产生新的长期 Memory
   ↓
Memory Write / Update
```

这套流程是理解 LangGraph Agent Memory、持久化 State、长期记忆、RAG 和 Scope 的基础。
