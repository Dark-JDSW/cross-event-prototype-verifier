# dsh-prompt-optimizer 0.6 安装记录

安装对象：<https://github.com/WestFox-AwA/dsh-prompt-optimizer> · Release **v0.6.10** · 包名 `@dsh-external/dsh-po06`
宿主：`@deepseek-ai/dsh` **0.1.7-rc.1**（升级前 0.1.5-rc.3）

## 一、结论一览（安装已完成并生效）

| 项 | 状态 |
|---|---|
| 发行包下载 + SHA256 校验 | ✅ `e3111c9bfb56c361cb172e3019d47a5e48174be7a0d33117d7c1e6f803d652c0`，与 Release 资产摘要一致 |
| 装入 profile `web` | ✅ `~\.dsh\profiles\web\node_modules\@dsh-external\dsh-po06`（0.6.10） |
| 注册为 bundle 层 | ✅ `dsh --profile web --dump-config` 组合结果末层出现 `- id: dsh-po06` |
| 作者自检 `check-install.mjs` | ✅ 4/4 通过，33 个 lib 文件与仓库逐字节一致 |
| 宿主升级到 0.1.7-rc.1 | ✅ 用 `scripts/upgrade-dsh.ps1` 完成（退出 → 卸载 → 安装 → 校验） |
| 宿主侧运行时 | ✅ `GET /po06/api/status` → 200，`version 0.6.10` |
| 客户端 UI | ✅ 已挂上：台账里出现 `trigger=user-message`，这条路径只有浏览器端面板在真实输入时才走 |
| 启用状态 | ✅ `enabled: true` + `rollout.mode: "all"`（`~\.dsh\po06.json`） |

## 一点五、本地补丁：配置变更即失效（**重装插件会被覆盖**）

`~\.dsh\profiles\web\node_modules\@dsh-external\dsh-po06\lib\index.js` 被本机改过**一处**：

- 新增 `import { …, watch } from 'node:fs'`；
- 在 `adapter.enableGate = createEnableGate(...)` 之后加了配置文件监听：`po06.json` 一变就
  `enableGate.invalidateAll()`，下一条消息立即重判。

**为什么**（真机实测）：`createEnableGate` 的结论带 **5 分钟保质期**，而"用户直接编辑配置文件"
不算"已知发生变化"——只有面板的 `POST /settings` 才会 `invalidateAll`。实测 20:36:14 写入
`enabled:true`，20:38:31 的台账仍是 `gate:rollout-off`，用户看到的就是**"装了却没反应"**。
补丁把最长 5 分钟的无效窗口压成 0（已用假 ctx 实测：改文件后结论表立即清空）。

**代价与维护**：这是宿主环境的本地补丁，**升级或重装插件会覆盖它**，需要按上面的两处重新写入。
上游若已修（`lib/rollout.js` 第 63–72 行那段注释说明作者知道 `gate:rollout-off` 这个问题），
请以官方版本为准。


## 二、0.1.5 → 0.1.7 这一步到底改变了什么

插件声明的客户端注入是 `["@deepseek-ai/dsh-client-runtime", "@deepseek-ai/dsh-client-ui-slots"]`，
这两个包**在 0.1.5-rc.3 与 0.1.7-rc.1 里都不存在**——它们其实是**陈旧声明**：插件的 `lib/client.js`
里没有任何一处 `require` 它们，真正起作用的是 `exports.inject = ['slots', 'locale']`，靠宿主注入 `slots` 服务。

所以关键不是那两个包，而是 **`slots` 服务由谁提供**：

- 0.1.5-rc.3：没有任何包提供 `slots`（`dsh-client-ui-renderer` 当时不提供）⇒ 面板必然挂不上。
- **0.1.7-rc.1：`@deepseek-ai/dsh-client-ui-renderer` 提供 `slots`**（`provideRoot` 等 API 在 `lib/client.js`）
  ⇒ 面板的 `exports.inject = ['slots','locale']` 得以满足。

也就是说，**升级宿主是这一步的真正解**，不需要额外去 registry 装 `dsh-client-ui-slots`
（那个包是纯核心库，没有 `dsh.client` 声明，根本进不了客户端模块图）。

插件注册的三个挂载点保留在 `lib/client.js`：

```js
mount('conversation.input.left', 'prompt-optimizer', 20, ControlBar)   // 输入栏左侧控件
mount('shell.overlay', NS + '-panel', 40, () => null)                 // 浮层占位
mount('settings.plugins.tab', NS, 40, SettingsTab)                    // 设置页一页
```

## 三、启用 / 关闭（**注意：面板里没有"总开关"**）

- **总开关只能写配置**：面板上的控件是 `assist`（自动/审阅/关闭）、`permission`、`detail`、
  `budget`、`model`、`historyMode`、`turns`、`readTools` —— 它们都**不写 `enabled`**。
- 启用闸门（`lib/assembly-gate.js`）只认带 `settingsVersion` 且 `enabled: true` 的配置，
  **拿不到就按保守方向不启用**（宁可少生效一轮，也不冒双重拦截的风险）。
- 当前 `~\.dsh\po06.json`（已写好，UTF-8 **无 BOM**）：

```json
{
  "settingsVersion": 1,
  "assist": "auto",
  "detail": "detailed",
  "budget": "generous",
  "model": null,
  "permission": "auto",
  "historyMode": "turns",
  "turns": 6,
  "readTools": false,
  "enabled": true,
  "rollout": { "mode": "all" }
}
```

- `rollout` 建议**显式写出来**。不写也能用（显式 `enabled:true` 时回落按 `all` 处理，
  见 `lib/rollout.js` 第 63–72 行），但显式写能避免台账里出现容易误读的
  `gateReason: invalid-or-missing-mode-defaults-to-off`。

- 关闭：把 `enabled` 改成 `false`（或删掉整个文件）。`assist: "off"` 是另一层"不拦截"，两者独立。
- **生效延迟**：启用判定有 **5 分钟保质期**（`createEnableGate` 的 `ttlMs = 5*60*1000`）。
  改完配置后，旧结论到期才重判；到期后的第一条消息就会带上意图包。想立刻生效就重启 DSH。
- `settingsVersion` 这个标记是刻意的：0.5.x 写在 `prompt-optimizer.json` 里的 `enabled: true`
  **不会**冒充 0.6 的启用意图。
- **不要写带 BOM 的 UTF-8**：作者真机踩过——`JSON.parse('\ufeff{…}')` 直接抛，
  配置会被判成 `config-unparsable` 而**静默不启用**。PowerShell 的 `Set-Content -Encoding utf8`
  会写出 BOM，用 `[System.IO.File]::WriteAllText($p, $json, (New-Object System.Text.UTF8Encoding($false)))`。

## 四、怎么自查它到底有没有在工作

```powershell
# 宿主侧：状态 + 闸门结论分布
Invoke-WebRequest http://127.0.0.1:3080/po06/api/status -UseBasicParsing | Select-Object -Expand Content

# 真机台账（每次拦截/投递一行）
Get-Content "$env:USERPROFILE\.dsh\po06-wire.jsonl" | Select-Object -Last 10
```

台账字段看两个就够了：

| 现象 | 含义 |
|---|---|
| `reason=gate-disabled` | 启用判定说"不启用"——检查 `po06.json` 的 `enabled` 与 5 分钟保质期 |
| `reason=old-plugin-unknown` | 拿不到 per-agent 作用域 ⇒ 保守不启用（装了 0.5.x 旧插件时也会撞 `DOUBLE_INTERCEPT`） |
| `ok=true` + `rounds/calls` | 本轮真的注入了意图包、工具循环跑起来了 |

## 五、安装落点（便于日后升级/回滚）

| 路径 | 作用 |
|---|---|
| `~\.dsh\profiles\web\node_modules\@dsh-external\dsh-po06\` | 插件本体（DSH 从这里解析第三方 bundle） |
| `~\.dsh\profiles\web\.dsh-po06-src\dsh-external-dsh-po06-0.6.10.tgz` | 发行包原件，供 pnpm 重装/升级用 |
| `~\.dsh\profiles\web\package.json` | `dependencies` 登记 + `dsh.profile.bundles` 注册 |
| `~\.dsh\po06.json` | 启用意图 + 面板设置（0.6 自己的文件） |
| `~\.dsh\po06-wire.jsonl` | 真机台账 |

> 注：本次没有跑 `pnpm add`——DSH 解析第三方 bundle 只认「profile 自己的 `node_modules`」，
> 手工放置即可生效。日后升级插件版本时走官方写法：`dsh plugin --profile web add <新的 tgz>`。

## 六、复现校验

```powershell
dsh --version                                                  # 期望 0.1.7-rc.1
dsh --profile web --dump-config | Select-String 'dsh-po06'     # 应出现 - id: dsh-po06
pwsh -NoProfile -File "<仓库>\scripts\upgrade-dsh.ps1"          # 版本与 profile 组合的复检
```

## 七、卸载

1. 从 `~\.dsh\profiles\web\package.json` 的 `dsh.profile.bundles` 与 `dependencies` 里移除 `@dsh-external/dsh-po06`；
2. 删掉 `~\.dsh\profiles\web\node_modules\@dsh-external\`；
3. 重启 DSH。配置 `~\.dsh\po06.json` 与台账 `po06-wire.jsonl` 可留可删。
