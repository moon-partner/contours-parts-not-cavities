# Submission package: GitHub + figshare DOI（中英双语步骤）

> ① 公开仓库（别人直接拿来用）② figshare 存档 DOI（可引用、免费、可预留后回填）。
> **平台变更**：原计划的 Zenodo 因网络不可达，已改为 figshare.com（流程几乎相同：
> 草稿 → Reserve DOI → 回填 → Publish）；国内打不开时备选 ScienceDB（科学数据银行）。
> 全程免费，约 30–40 分钟。

---

## 第 0 步：作者名（✅ 已完成）

已统一为 `Fan, Tianyu`：`CITATION.cff`（`family-names: Fan` / `given-names: Tianyu` /
`affiliation: Independent researcher`）与 `zenodo/metadata.json`
（`creators[0].name: "Fan, Tianyu"`），zip 已随之重打。

## 第 1 步：GitHub 公开仓库（10 分钟）

1. 网页端 github.com → 右上角 ＋ → **New repository**；
   - 名字建议 `contours-parts-not-cavities`；
   - 选 **Public**；⚠️ 不要勾 "Add a README file" 和 ".gitignore"（仓库自带，会冲突）。
2. 本地 PowerShell：

```powershell
cd E:\Blender\release
git config --global user.name  "Fan Tianyu"      # 首次必填，否则 commit 报错
git config --global user.email "you@example.com"
git init
git add -A
git commit -m "First release: verified artifacts (98/98 assertions)"
git branch -M main
git remote add origin https://github.com/<你的用户名>/contours-parts-not-cavities.git
git push -u origin main
```

3. push 时 Windows 弹 Git Credential Manager 浏览器登录窗口 → 登录 GitHub → Authorize。
   若被网络挡住：GitHub → Settings → Developer settings → Personal access tokens (classic)
   → Generate new token（勾 repo），凭据窗口用户名填 GitHub 用户名、密码粘贴 token。
4. 建好后回填仓库 URL（两处）：
   - `zenodo/metadata.json` → `related_identifiers[0].identifier`（替换 `TODO_GITHUB_URL`）；
   - `CITATION.cff` → `repository-code`（替换 TBD 行）；然后 `git add -A && git commit`。

## 第 2 步：figshare 上传 + Reserve DOI（15 分钟）

1. 打开 <https://figshare.com> → Sign Up / Log in（邮箱或 ORCID 均可）；
2. 登录后点 **New upload**（或头像菜单里的 Upload）→ 把 `E:\Blender\zenodo_upload.zip`
   （约 9MB）整体拖进上传区——zip 作为归档附件保留（不解包正常；可浏览版在 GitHub）；
3. 表单逐字段对照 `zenodo/metadata.json` 填写：

| 表单字段 | 填什么 |
|---|---|
| Title | `title` 整段 |
| Item type | Software（没有就选 Text / Other） |
| Creators | `Fan, Tianyu`，类型 Individual，Affiliation `Independent researcher` |
| Description | `zenodo/DESCRIPTION.txt` 纯文本整段（别粘 HTML） |
| License | 搜选 MIT（若无 MIT → CC BY 4.0，并在描述末注明代码 MIT） |
| Keywords | `keywords` 9 个（3D reconstruction, visual hull, identifiability, silhouettes, part assembly, skeleton topology, SDF, solid prior, reproducibility） |
| Version | 1.0.0 |
| Related works | GitHub 仓库 URL，关系选 `Is supplement to`（没有就 `Is related to`） |
| Language | English（若有该字段） |

4. 草稿页找 **Reserve DOI**（一般在标题下方或右侧面板；找不到就先保存草稿再刷新）→
   得到 `10.6084/m9.figshare.XXXXXXX`。**此时先不要点 Publish**。

### 备选：ScienceDB（figshare 打不开时）

<https://www.scidb.cn> → 手机号注册 → 实名认证 → 创建数据集 → 上传同一个 zip →
字段同上表（英文填）→ 提交审核（1–3 天）→ 得到 `10.57760/...` DOI。DOI 为 DataCite 注册，
国际同样可检索。拿到 DOI 后回填步骤与第 3 步完全相同。

## 第 3 步：回填 DOI → 发布 → 推送（5 分钟）

1. DOI 回填两处（`XXXXXXX` 换成真实号码）：
   - `CITATION.cff`：`version: 1.0.0` 下面加一行 `doi: 10.6084/m9.figshare.XXXXXXX`；
   - `README.md` Cite 段：`See CITATION.cff. DOI: 10.6084/m9.figshare.XXXXXXX`。
2. 重打 zip 并替换草稿里的文件（让发布的快照自带 DOI）：

```powershell
Compress-Archive -Path E:\Blender\release\* -DestinationPath E:\Blender\zenodo_upload.zip -Force
```

3. figshare 草稿页 → 删除旧 zip → 拖入新 zip → **Publish**
   （发布后条目不可修改，日后要改走新版本）；
4. 回 GitHub：`git add -A && git commit -m "Add reserved DOI" && git push`。
5. 验收：figshare 页面显示 DOI、Cite 按钮（APA/BibTeX 一键复制）、Files 里有 zip；
   GitHub README 的 Cite 段带 DOI。

## 第 4 步（可选）：让别人更容易找到它

- GitHub 仓库 → Releases → 新建 `v1.0.0` release（附 zip）；
- 把仓库提交到 <https://paperswithcode.com>（有提交入口）；
- `ZONG_REPORT.md` 可作为中文版说明留在仓库。

## 备注

- **arXiv 不作为默认选项**：独立身份首发常卡在 endorsement 制度上；figshare DOI 同样可引用、
  免费、无门槛。若以后有合作者能背书，再迁 arXiv 不迟。
- 投稿（AAAI-27 workshop 等）与本存档**互不冲突**：非存档 workshop 均接受已发预印本的稿件。
- 上传失败常见原因：文件名含中文（包内 `总报告.md` 已改名 `ZONG_REPORT.md`）、zip 超 50GB
  （我们仅 9MB，无虞）。
