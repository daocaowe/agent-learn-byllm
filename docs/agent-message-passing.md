# Agent 开发中的消息传递机制：State、Memory、Context、Message、Checkpoint

> 学习主题：理解 Agent / LangGraph 中几个核心状态与消息概念之间的关系，以及它们如何共同完成多轮对话、Tool Calling、状态恢复和 Multi-Agent 消息传递。

## 1. 整体关系

Agent 开发中的“消息传递”并不只是 Message。更完整的关系可以理解为：

```text
                    用户请求
                       │
                       ▼
                  ┌─────────┐
                  │ Message │
                  │ 用户消息 │
                  └────┬────┘
                       │
                       ▼
              ┌─────────────────┐
              │      State      │
              │ 当前 Agent 状态 │
              │ messages        │
              │ user_id         │
              │ intent          │
              │ retrieved_docs  │
              │ tool_result     │
              └───────┬─────────┘
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
       Context      Tool        LLM 调用
      当前上下文    工具结果       │
          │           │           │
          └───────────┼───────────┘
                      ▼
                 State 更新
                      │
                      ▼
                Checkpoint
              保存当前 State
                      │
              ┌───────┴───────┐
              ▼               ▼
           下一轮           程序恢复
              │               │
              ▼               ▼
           Memory         Checkpoint
         长期/短期记忆      恢复状态
```

最重要的一句话：

> **Message 是“发生了什么”，State 是“Agent 现在处于什么状态”，Context 是“这次调用需要提供什么信息”，Memory 是“过去保存了什么信息”，Checkpoint 是“某个时间点 State 的快照”。**

---

## 2. Message：消息

Message 是 Agent 世界中最基础的数据交换单位。

LangChain 中常见的消息类型包括：

- `SystemMessage`：系统指令
- `HumanMessage`：用户消息
- `AIMessage`：模型消息
- `ToolMessage`：工具执行结果

示例：

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="你是一个专业的 AI 助手"),
    HumanMessage(content="帮我查询北京天气"),
]
```

典型消息链：

```text
SystemMessage
      ↓
HumanMessage
      ↓
     LLM
      ↓
AIMessage
```

如果发生 Tool Calling：

```text
User Message
      ↓
    Agent
      ↓
     LLM
      ↓
 AIMessage（包含 tool_calls）
      ↓
     Tool
      ↓
 ToolMessage
      ↓
     LLM
      ↓
最终 AIMessage
```

### 一个完整的 Tool Calling 消息序列

```python
[
    HumanMessage(
        content="北京天气怎么样？"
    ),

    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {
                    "city": "北京"
                }
            }
        ]
    ),

    ToolMessage(
        content="北京：晴，32℃"
    ),

    AIMessage(
        content="北京今天晴天，32℃。"
    )
]
```

因此，**Message 主要负责表达 Agent 之间、LLM 与 Tool 之间发生了什么。**

---

## 3. State：Agent 的当前状态

State 比 Message 更高一层。

可以把 State 理解成：

> **Agent 当前这一轮任务的“状态对象”。**

例如一个 RAG Agent：

```python
from typing import TypedDict
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    messages: list[AnyMessage]
    user_id: str
    intent: str
    query: str
    documents: list[str]
    tool_result: str
    answer: str
```

State 可能是：

```python
state = {
    "messages": [
        HumanMessage(content="公司的年假制度是什么？")
    ],
    "user_id": "10001",
    "intent": "rag",
    "query": "公司的年假制度",
    "documents": [
        "员工工作满一年享受 5 天年假..."
    ],
    "tool_result": "",
    "answer": ""
}
```

### State 与 Message 的关系

**Message 是 State 的一个字段，而 State 不等于 Message。**

```text
State
│
├── messages
│     ├── HumanMessage
│     ├── AIMessage
│     ├── ToolMessage
│     └── AIMessage
│
├── user_id
├── intent
├── documents
├── tool_result
└── answer
```

所以：

```text
State
 ├── messages
 ├── user_id
 ├── intent
 ├── documents
 ├── tool_result
 └── answer
```

而：

```text
messages
 ├── HumanMessage
 ├── AIMessage
 ├── ToolMessage
 └── AIMessage
```

---

## 4. Context：当前调用需要的信息

Context 可以理解成：

> **为了完成当前任务，需要临时提供给 Agent / LLM / Tool 的信息。**

例如：

```python
context = {
    "user_id": "10001",
    "current_time": "2026-08-16",
    "user_role": "admin",
    "language": "zh-CN",
}
```

Context 和 Memory 不一样：

```text
Context
    ↓
当前请求需要的信息
```

例如：

```text
当前用户是管理员
```

而 Memory：

```text
过去保存的信息
```

例如：

```text
用户以前喜欢使用 Markdown 回答
```

### Context 的安全边界

`user_id`、`tenant_id`、权限、角色等可信信息，不应该完全交给 LLM 自己生成。

例如：

```text
LLM 生成的数据
        ≠
可信系统 Context
```

正确方式应该是从系统侧的 Context 获取：

```python
user_id = context["user_id"]
```

而不是让模型自己决定用户 ID。

---

## 5. Memory：跨轮次记忆

Memory 解决的是：

> **上一轮发生的事情，下一轮还能不能记住？**

例如：

### 第一轮

```text
用户：我叫张三，我正在开发一个 RAG 系统。
Agent：好的，你正在开发 RAG 系统。
```

### 第二轮

```text
用户：那我的项目怎么增加 Agent 能力？
```

如果存在 Memory，Agent 可以知道：

```text
张三
 ↓
正在开发 RAG 系统
 ↓
现在想增加 Agent 能力
```

### 短期记忆

短期记忆通常与当前 conversation / thread 相关：

```text
conversation_id = abc123

第 1 轮：用户：我叫张三
第 2 轮：用户：我正在做 RAG
第 3 轮：用户：怎么增加 Agent？
第 4 轮：用户：刚才说的 Tool 怎么实现？
```

这些信息可以体现在：

```python
state["messages"]
```

并通过 Checkpoint 持久化。

### 长期记忆

长期记忆一般独立于某一个具体对话，例如：

```text
user_id = 10001

用户长期信息：
- 使用 Python
- 使用 LangChain
- 正在学习 Agent
- 喜欢代码示例
```

可以使用：

- PostgreSQL
- Redis
- MongoDB
- Vector DB
- 其他专用 Memory Store

例如：

```python
memory = {
    "user_id": "10001",
    "preferences": {
        "language": "Python",
        "style": "代码示例优先"
    }
}
```

---

## 6. Checkpoint：State 的快照

Checkpoint 可以理解为：

> **Agent 执行过程中某个时间点的 State 快照。**

例如 Agent 执行：

```text
用户问题
   ↓
Intent
   ↓
RAG
   ↓
Tool
   ↓
LLM
   ↓
Answer
```

在不同节点可以保存不同的状态：

```text
State V1
   ↓
Checkpoint 1
   ↓
State V2
   ↓
Checkpoint 2
   ↓
State V3
```

### Checkpoint 的价值

Agent 经常不是一次执行完成，例如：

```text
用户：帮我查询订单，然后退款

Step 1：查询订单
        ↓
Step 2：确认订单
        ↓
Step 3：调用退款 Tool
        ↓
Step 4：等待人工确认
```

如果 Step 3 之后服务器挂掉：

- 没有 Checkpoint：状态可能丢失
- 有 Checkpoint：可以恢复到之前保存的 State 并继续执行

因此 Checkpoint 的核心能力包括：

```text
持久化状态
    +
恢复执行
    +
Human-in-the-loop
    +
状态历史 / 时间旅行
```

---

## 7. 五个概念对比

| 概念 | 核心作用 | 生命周期 | 典型内容 |
|---|---|---|---|
| Message | 表达消息事件 | 单条消息 / 消息序列 | User、AI、Tool 消息 |
| State | 当前 Agent 状态 | 当前执行过程 | messages、intent、docs、answer |
| Context | 当前调用需要的信息 | 当前调用 | user_id、权限、时间、租户 |
| Memory | 保存过去的信息 | 跨轮次 / 长期 | 用户偏好、历史事实 |
| Checkpoint | State 快照 | 持久化 | 某一时刻完整 Agent 状态 |

可以简单记成：

```text
Message     = 说了什么
State       = 现在是什么状态
Context     = 这次需要知道什么
Memory      = 以前记住了什么
Checkpoint  = 某一时刻保存了什么状态
```

---

## 8. LangGraph 示例：State + Message + Checkpoint

下面用一个简单的订单 Agent 演示。

### 8.1 定义 State

```python
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import HumanMessage, AIMessage


class AgentState(TypedDict):
    messages: list
    order_id: str
    order_status: str
    answer: str
```

### 8.2 查询订单 Node

```python
def query_order(state: AgentState):
    order_id = state["order_id"]

    # 模拟查询数据库
    order_status = "已支付，未发货"

    return {
        "order_status": order_status
    }
```

### 8.3 LLM Node

```python
def llm_node(state: AgentState):
    messages = state["messages"]
    order_status = state["order_status"]

    answer = f"""
订单 {state['order_id']} 当前状态：

{order_status}

由于订单尚未发货，因此可以申请退款。
"""

    return {
        "messages": messages + [
            AIMessage(content=answer)
        ],
        "answer": answer
    }
```

### 8.4 构建 Graph

```python
builder = StateGraph(AgentState)

builder.add_node("query_order", query_order)
builder.add_node("llm", llm_node)

builder.add_edge(START, "query_order")
builder.add_edge("query_order", "llm")
builder.add_edge("llm", END)
```

### 8.5 添加 Checkpoint

```python
checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)
```

### 8.6 执行 Agent

```python
result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="帮我查询订单 123 能不能退款"
            )
        ],
        "order_id": "123",
        "order_status": "",
        "answer": ""
    },
    config={
        "configurable": {
            "thread_id": "user_10001"
        }
    }
)
```

这里的 `thread_id` 很重要，可以把它理解为：

```text
conversation_id
```

或者：

```text
Agent 执行线程 ID
```

Checkpoint 可以根据 `thread_id` 保存和恢复对应的 State。

---

## 9. 第二次请求：短期记忆如何出现

用户继续：

```text
那如果已经发货了呢？
```

继续使用相同的 `thread_id`：

```python
graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="那如果已经发货了呢？"
            )
        ],
        "order_id": "123",
        "order_status": "",
        "answer": ""
    },
    config={
        "configurable": {
            "thread_id": "user_10001"
        }
    }
)
```

Checkpoint 可以找到之前线程对应的状态，从而实现类似短期对话记忆的效果：

```text
第一次
─────────────────
State
 ├── messages
 │    ├── User
 │    └── AI
 ├── order_id=123
 └── order_status=已支付，未发货

          ↓

Checkpoint
          ↓

第二次
─────────────────
恢复 State

 ├── messages
 │    ├── User
 │    ├── AI
 │    └── User：那如果已经发货了？
 │
 └── order_id=123
```

---

## 10. Agent Tool Calling 中的消息传递

典型流程：

```text
User
 ↓
HumanMessage
 ↓
LLM
 ↓
AIMessage
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"city": "北京"}
        }
    ]
 ↓
Tool
 ↓
ToolMessage
 ↓
LLM
 ↓
AIMessage
```

因此 Tool Calling 本质上也是一种消息传递协议：

```text
AIMessage
    ↓
Tool Call
    ↓
Tool 执行
    ↓
ToolMessage
    ↓
LLM
```

---

## 11. Agent 内部的三层消息传递

### 第一层：Agent 内部 Message Passing

```text
User
 ↓
LLM
 ↓
Tool
 ↓
LLM
 ↓
Agent
```

通过：

```text
HumanMessage
AIMessage
ToolMessage
SystemMessage
```

传递信息。

### 第二层：Node → Node

LangGraph 中：

```text
Node A
  ↓
State
  ↓
Node B
  ↓
State
  ↓
Node C
```

例如：

```text
Intent Node
     ↓
intent
     ↓
Retriever Node
     ↓
documents
     ↓
LLM Node
     ↓
answer
```

这里主要依赖 `State` 传递。

### 第三层：Agent → Agent

Multi-Agent 中：

```text
              Supervisor
                  │
        ┌─────────┼─────────┐
        ↓         ↓         ↓
      RAG Agent  SQL Agent  Search Agent
        │         │         │
        └─────────┼─────────┘
                  ↓
              Supervisor
```

可以抽象成：

```python
{
    "from": "supervisor",
    "to": "rag_agent",
    "task": "查询公司年假制度",
    "context": {
        "user_id": "10001"
    }
}
```

RAG Agent 返回：

```python
{
    "from": "rag_agent",
    "to": "supervisor",
    "result": "员工满一年享受 5 天年假"
}
```

这就是更高级的 **Agent-to-Agent Message Passing**。

---

## 12. Memory 与 Context 的关系

实际项目中经常是：

```text
                    User Request
                         │
                         ▼
                  ┌─────────────┐
                  │   Memory    │
                  └──────┬──────┘
                         │
                    加载历史信息
                         │
                         ▼
                  ┌─────────────┐
                  │   Context   │
                  └──────┬──────┘
                         │
                         ▼
                       Agent
```

例如 Memory 中保存：

```python
memory = {
    "user_name": "张三",
    "project": "RAG 系统",
    "preference": "回答尽量代码化"
}
```

当前请求可能只需要：

```python
context = {
    "user_name": "张三",
    "project": "RAG 系统"
}
```

然后：

```text
Context
   ↓
Prompt
   ↓
LLM
```

因此：

> **Memory 是长期信息的存储机制，Context 是这些信息在当前任务中的工作集。**

---

## 13. 一个完整的 Agent 架构图

```text
                         ┌──────────────┐
                         │ Long Memory  │
                         │ 用户长期记忆 │
                         └──────┬───────┘
                                │
                                ▼
User ──Message──────────────► Context
                                │
                                ▼
                        ┌───────────────┐
                        │     State     │
                        │               │
                        │ messages      │
                        │ intent        │
                        │ query         │
                        │ documents     │
                        │ tool_result   │
                        │ answer        │
                        └───────┬───────┘
                                │
                     ┌──────────┼──────────┐
                     ▼          ▼          ▼
                    LLM        Tool      Retriever
                     │          │          │
                     └──────────┼──────────┘
                                ▼
                             Message
                                │
                                ▼
                             State
                                │
                                ▼
                         ┌─────────────┐
                         │ Checkpoint  │
                         │ State 快照  │
                         └──────┬──────┘
                                │
                          下一轮 / 恢复
                                │
                                ▼
                              Agent
```

---

## 14. 最后记忆方法

推荐直接记住这条链：

```text
Message
   ↓
State
   ↓
Context
   ↓
LLM / Tool
   ↓
Message
   ↓
State 更新
   ↓
Checkpoint
```

而 Memory 的作用是：

```text
Memory
   ↓
为下一次任务提供历史信息
   ↓
形成新的 Context
```

最终可以浓缩成：

```text
Message     = 说了什么
State       = 现在是什么状态
Context     = 这次需要知道什么
Memory      = 以前记住了什么
Checkpoint  = 某一时刻保存了什么状态
```

---

## 15. 下一步学习顺序

如果继续深入 LangGraph / Agent 消息传递机制，建议按下面顺序学习：

```text
1. LangGraph State Reducer
        ↓
2. Message Graph / ToolMessage
        ↓
3. Checkpoint + Thread + Interrupt
        ↓
4. Multi-Agent Message Passing
```

其中重点关注：

- **State Reducer**：理解多个 Node 如何合并和更新 State
- **Message / ToolMessage**：理解 LLM 与 Tool 的标准消息流
- **Checkpoint + Thread**：理解多轮对话和状态持久化
- **Interrupt / Human-in-the-loop**：理解人工确认与暂停恢复
- **Multi-Agent Message Passing**：理解 Supervisor、Worker Agent 之间如何传递任务和结果

这些内容是理解 Agent 为什么能够实现“多步骤执行、循环调用 Tool、人工介入、失败恢复、多 Agent 协作”的基础。
