# Agent Memory System：短期记忆、工作记忆、长期记忆与外部记忆

> 本文整理本次 Agent 开发学习中的 Memory System 内容，并以“自助台球客服 Agent”为完整案例。

## 1. 核心结论

Agent 的记忆不能简单理解为 Chat History。工程上至少可以拆成四层：

| 类型 | 主要解决的问题 | 典型内容 | 生命周期 |
|---|---|---|---|
| 短期记忆 | 当前对话聊了什么 | messages / conversation history | 当前 Session / Thread |
| 工作记忆 | 当前任务做到哪一步 | State、任务参数、工具结果、中间状态 | 当前任务 |
| 长期记忆 | 跨 Session 还记得用户什么 | 用户偏好、长期事实、目标 | 持久化 |
| 外部记忆 | Agent 可以从外部世界查到什么 | RAG、知识库、业务文档 | 持久化 |

最重要的一点：

> Memory 的分类由信息的语义、用途和生命周期决定，而不是由 Redis、PostgreSQL、Qdrant 等存储介质决定。

例如 Redis 可以存短期状态，PostgreSQL 也可以存短期状态；长期 Memory 可以用 PostgreSQL 保存，也可以结合向量检索。

---

# 2. 自助台球客服 Agent 场景

假设有一家无人台球厅，用户可以通过 App / 微信和 Agent 对话。

例如：

```text
用户：帮我订今晚8点的2号桌。
Agent：好的，请问打几个小时？
用户：2小时。
```

Agent 可以调用：

```text
check_table_availability()   查询球桌是否可用
query_membership()           查询会员等级和折扣
query_price()                查询球桌价格
book_table()                 创建预约
cancel_booking()             取消预约
```

系统可以进一步包含：

```text
短期记忆       LangGraph Checkpointer
工作记忆       LangGraph AgentState
长期记忆       LangGraph Store / PostgreSQL
外部记忆       Qdrant / Vector DB / Knowledge Base
```

---

# 3. 短期记忆：Conversation History

## 3.1 基本概念

用户说：

```text
User：我要订2号桌。
Assistant：好的，请问什么时候？
User：今晚8点。
Assistant：好的，请问打几个小时？
User：2小时。
```

Agent 必须知道最后的“2小时”对应的是前面的预约任务。

概念上就是：

```python
messages = [
    HumanMessage("我要订2号桌"),
    AIMessage("好的，请问什么时候？"),
    HumanMessage("今晚8点"),
    AIMessage("好的，请问打几个小时？"),
    HumanMessage("2小时"),
]
```

LLM 本身并不是自动永久记住了这些内容，而是应用在请求中提供上下文。

## 3.2 LangGraph 中的短期记忆

常见做法是：

```text
messages
   +
checkpointer
   +
thread_id
```

例如：

```python
config = {
    "configurable": {
        "thread_id": "user_001_session_001"
    }
}
```

第一次请求：

```text
thread_id = user_001_session_001
用户：我要订2号桌
```

下一次仍然使用同一个 thread：

```text
thread_id = user_001_session_001
用户：今晚8点
```

LangGraph 可以恢复之前的 State，因此 Agent 知道“今晚8点”是在补充 2 号桌的预约信息。

## 3.3 短期记忆的生命周期

```text
Session A
  ↓
Conversation History
  ↓
当前任务完成
  ↓
Session 结束 / 压缩 / 归档
```

所以：

> 短期记忆主要解决“这次对话发生了什么”。

---

# 4. 工作记忆：Working Memory

短期记忆和工作记忆很容易混淆。

假设用户直接说：

```text
帮我订今晚8点的2号桌，打两个小时。
```

Agent 可能执行：

```text
Step 1：解析需求
Step 2：查询 2 号桌 20:00 是否可用
Step 3：查询用户会员等级
Step 4：查询价格
Step 5：计算折扣价格
Step 6：创建订单
Step 7：返回订单号
```

此时 Agent 需要一份结构化的当前任务状态：

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

这就是 Working Memory。

可以把它理解成：

> Agent 当前脑子里的“工作台 / 草稿纸”。

---

# 5. Conversation History vs Working Memory

推荐这样区分：

| | Conversation History | Working Memory |
|---|---|---|
| 形式 | messages | 结构化 State |
| 内容 | 用户/Agent/Tool 原始消息 | 当前任务的重要字段 |
| 作用 | 理解对话上下文 | 驱动任务执行 |
| 例子 | “我要订2号桌” | `table_id=2` |
| 生命周期 | 当前 Thread | 当前任务 |

例如：

```text
Conversation History
────────────────────────
用户：我要订2号桌
用户：今晚8点
用户：2小时
```

经过 Agent 处理后：

```text
Working Memory
────────────────────────
task = booking
table_id = 2
booking_time = 20:00
duration = 2
```

所以：

> Conversation History 更像“原始记忆”，Working Memory 更像“整理后的任务状态”。

---

# 6. 工具调用与 Working Memory

这是 Agent 开发中非常重要的一点。

Agent 调用：

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

工具结果不能只看完就丢掉，而应该参与下一轮 Agent 推理：

```text
Tool
 ↓
Tool Result
 ↓
Working Memory / State
 ↓
下一次 Agent 推理
```

例如：

```python
state["available"] = True
```

然后查询会员：

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

更新：

```python
state["membership_level"] = "VIP"
state["discount"] = 0.8
```

于是 Agent Loop 变成：

```text
User
 ↓
Agent / LLM
 ↓
决定 Tool
 ↓
Tool
 ↓
Tool Result
 ↓
Working Memory / State
 ↓
Agent / LLM
 ↓
决定下一步 Tool
 ↓
Tool
 ↓
Tool Result
 ↓
Working Memory / State
 ↓
...
 ↓
Final Answer
```

这就是 Agent 与普通 Chatbot 的核心区别之一。

---

# 7. AgentState 示例

LangGraph 中可以把 Working Memory 放在 State：

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

    # 工具查询结果
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

这里：

```text
messages
```

主要承担 Conversation History；

而：

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

主要承担 Working Memory。

---

# 8. 长期记忆：Long-term Memory

假设用户说：

```text
我是张三，我经常周末晚上来打台球。
```

这条信息未来 Session 可能仍然有价值，可以提取为长期记忆：

```json
{
  "user_id": "123",
  "memory": "用户经常周末晚上打台球",
  "type": "preference",
  "scope": "global"
}
```

第二周用户重新进入 App，开启了一个新的 Session，Agent 仍然可以读取这条信息。

## 8.1 长期记忆不等于保存所有聊天

不应该：

```text
所有历史消息
 ↓
全部永久保存成 Memory
```

例如：

```text
“我今天8点来打台球。”
```

通常没必要变成长期记忆。

而：

```text
“我一般周末晚上来打球。”
“我的后端项目都使用 FastAPI。”
“我通常一次打2小时。”
```

更适合长期保存。

## 8.2 Memory Write 生命周期

```text
Conversation
 ↓
Memory Extraction
 ↓
重要性判断
 ↓
Scope 判断
 ↓
去重 / 更新
 ↓
Persistent Store
```

可以设计 Memory：

```python
memory = {
    "user_id": "123",
    "content": "用户偏好 FastAPI",
    "type": "preference",
    "scope": "global",
    "importance": 0.9,
}
```

---

# 9. Global Scope 与 Project Scope

Scope 解决的问题是：

> 这条记忆到底对谁有效、在哪个范围内有效？

## 9.1 Global Scope

Global Memory 对用户所有项目都有效：

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

## 9.2 Project Scope

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

如果把“用户喜欢2号球桌”放到 Global，那么另一个咖啡店客服 Agent 也可能读取到它，产生 Memory Contamination。

---

# 10. LangGraph Store 的 Scope 实现

可以用 namespace 实现 Scope。

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

保存 Global Memory：

```python
memory_store.put(
    ("user", user_id, "global"),
    "preference_001",
    {
        "type": "preference",
        "content": "用户通常打2小时",
    },
)
```

保存 Project Memory：

```python
memory_store.put(
    ("user", user_id, "project", "billiard"),
    "billiard_001",
    {
        "type": "preference",
        "content": "用户喜欢2号球桌",
    },
)
```

读取：

```python
global_memories = memory_store.search(
    ("user", user_id, "global")
)

project_memories = memory_store.search(
    ("user", user_id, "project", project_id)
)
```

`InMemoryStore` 适合学习；生产环境应根据系统要求使用持久化 Store / 数据库。

---

# 11. 外部记忆：RAG / Knowledge Base

台球厅还存在大量业务知识：

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

这时候不是查询用户 Memory，而是查业务知识库：

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

## 11.1 Long-term Memory vs RAG

| Long-term Memory | RAG / External Memory |
|---|---|
| 记住用户 | 记住外部世界知识 |
| 用户偏好 | 公司文档 |
| 用户长期事实 | 产品说明 |
| 用户目标 | 规则、价格、API 文档 |
| 通常按用户隔离 | 通常按知识库 / Tenant / Project 隔离 |

一句话：

> **Memory 记住“你”，RAG 记住“世界”。**

---

# 12. Memory Read：什么时候读取

不要每次把全部 Memory 都塞进 Prompt：

```text
1000 条 Memory
 ↓
全部放 Prompt
 ↓
LLM
```

这会造成：

- Token 浪费
- 延迟增加
- 无关信息噪声
- 错误关联
- Context Window 压力

正确方向：

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

例如：

```text
用户：帮我订个位置。
```

可以检索：

```text
Global：用户通常打2小时
Project：用户喜欢2号桌
```

而不是把用户全部历史 Memory 都发送给模型。

所以：

> Memory System 本质上也包含 Retrieval。

---

# 13. Memory Write：什么时候写

一个典型设计：

```text
User Message
 ↓
Memory Extractor
 ↓
重要性判断
 ↓
Scope 判断
 ↓
去重 / 冲突处理
 ↓
Memory Store
```

例如：

```text
“以后我的项目都使用 FastAPI。”
```

可以保存：

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

# 14. 完整 Memory 架构

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

# 15. LangGraph 台球客服代码 Demo

## 15.1 Tools

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


tools = [
    check_table_availability,
    query_membership,
    query_price,
    book_table,
]
```

## 15.2 State

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

## 15.3 Long-term Memory

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

初始化：

```python
USER_ID = "user_001"
PROJECT_ID = "billiard"

save_global_memory(
    USER_ID,
    "preference_001",
    {
        "type": "preference",
        "content": "用户喜欢中文交流",
    },
)

save_global_memory(
    USER_ID,
    "preference_002",
    {
        "type": "preference",
        "content": "用户通常一次打2小时",
    },
)

save_project_memory(
    USER_ID,
    PROJECT_ID,
    "billiard_001",
    {
        "type": "preference",
        "content": "用户在台球项目中喜欢2号桌",
    },
)
```

此时：

```text
Global
├── 喜欢中文
└── 通常打2小时

Project: billiard
└── 喜欢2号桌
```

## 15.4 Memory Retrieval

```python
def retrieve_memories(user_id: str, project_id: str):
    global_memories = get_global_memories(user_id)
    project_memories = get_project_memories(
        user_id,
        project_id,
    )

    return {
        "global": global_memories,
        "project": project_memories,
    }
```

## 15.5 Agent Node

下面是概念性实现：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key="YOUR_API_KEY",
    base_url="https://api.deepseek.com",
)


def agent_node(state: AgentState):
    memories = retrieve_memories(
        state["user_id"],
        state["project_id"],
    )

    global_memory_text = "\n".join(
        item.value["content"]
        for item in memories["global"]
    )

    project_memory_text = "\n".join(
        item.value["content"]
        for item in memories["project"]
    )

    system_prompt = f"""
你是一个自助台球厅客服 Agent。

你的任务：
1. 帮用户查询桌位
2. 查询会员
3. 查询价格
4. 创建预约
5. 必要时调用工具

当前用户：{state["user_id"]}

Global Memory：
{global_memory_text}

Project Memory：
{project_memory_text}

当前工作状态：
 table_id={state.get("table_id")}
 booking_time={state.get("booking_time")}
 duration={state.get("duration")}
 available={state.get("available")}
 membership_level={state.get("membership_level")}
 price={state.get("price")}
 booking_id={state.get("booking_id")}

请根据当前状态决定下一步操作。
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ] + state["messages"]

    response = model.bind_tools(tools).invoke(messages)

    return {
        "messages": [response],
        "current_step": "agent",
    }
```

## 15.6 Tool Node

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

判断是否继续工具调用：

```python
from typing import Literal
from langgraph.graph import END


def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END
```

## 15.7 Graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
)
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

> 注意：上面的代码主要用于理解 Memory、State 和 Agent Loop。生产项目还需要增加参数校验、工具异常处理、权限校验、订单幂等、数据库事务、Memory 检索排序和敏感数据保护。

---

# 16. 完整预约过程中的 Working Memory 变化

用户：

```text
帮我订今晚8点的2号桌，打两个小时。
```

初始：

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

假设：

```text
hourly_price = 50
original_price = 2 × 50 = 100
```

计算：

```text
100 × 0.8 = 80
```

最后调用：

```text
book_table()
```

得到：

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

# 17. 四种 Memory 最终对照

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

# 18. Memory Lifecycle

真正工程化的 Memory 系统重点不是“会不会存”，而是完整生命周期：

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

因此成熟 Agent 可以抽象成：

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

# 19. 生产化架构升级

当前 Demo 使用 `InMemorySaver` / `InMemoryStore` 是为了学习概念。

进一步生产化可以演进为：

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

一种可能的职责划分：

```text
FastAPI
└── API / Authentication / Request Validation

LangGraph
└── Agent Orchestration / State / Tool Loop

Redis
└── 可选：高频 Session / Cache / TTL 数据

PostgreSQL
└── 用户、订单、长期 Memory、业务数据

Qdrant
└── RAG / Semantic Retrieval / 大规模知识库
```

注意：这不是强制绑定。真正工程设计应该根据一致性、TTL、查询方式、规模、成本和运维能力选择存储。

---

# 20. 学习重点

学习 Agent Memory 时重点掌握：

1. Conversation History
2. Working Memory / State
3. Long-term Memory
4. External Memory / RAG
5. Memory Write
6. Memory Read / Retrieval
7. Scope
8. Memory Lifecycle

三个必须记住的区别：

> **Memory ≠ Chat History**

> **Memory ≠ RAG**

> **Memory = 信息持久化 + 生命周期管理 + 按需检索 + Scope 管理**

---

# 21. 人脑类比

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

最终可以记住这一条完整链路：

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

这套模型是后续学习 LangGraph Agent Memory、持久化 State、长期记忆、RAG、Multi-Agent Context Sharing 和 Scope 隔离的基础。
