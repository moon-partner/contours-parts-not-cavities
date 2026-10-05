# Submission package: GitHub + Zenodo DOI（中英双语步骤）

> 目标：① 公开仓库（别人直接拿来用）② Zenodo 存档 DOI（可引用、免费、先预留后回填）。
> **进度**：第 0/1 步已完成；第 2/3 步进行中。Zenodo 若再次不可达，备选 figshare.com
> 或科学数据银行 scidb.cn（字段对照相同，DOI 前缀不同，回填步骤完全一样）。
> 全程免费，约 30–40 分钟。

---

## 第 0 步：作者名（✅ 已完成）

已统一为 `Fan, Tianyu`：`CITATION.cff`（`family-names: Fan` / `given-names: Tianyu` /
`affiliation: Independent researcher`）与 `zenodo/metadata.json`
（`creators[0].name: "Fan, Tianyu"`），zip 已随之重打。

## 第 1 步：GitHub 公开仓库（✅ 已完成）

- 仓库：<https://github.com/moon-partner/contours-parts-not-cavities>（Public，291 文件）
- 提交：`e0cb9c2 First release: verified artifacts (98/98 assertions)`（作者 Fan Tianyu）→
  `60bdd67 Add GitHub repository URL`
- 仓库 URL 已回填 `CITATION.cff` 的 `repository-code` 与 `zenodo/metadata.json` 的
  `related_identifiers`，并已推送。

## 第 2 步：Zenodo 上传 + Reserve DOI（15 分钟）

1. 打开 <https://zenodo.org> → **Log in** → 选 **Continue with GitHub** → 授权
   （用第 1 步的账号一键登录，无需另注册）；
2. 登录后点 **New upload**；
3. 把 `E:\Blender\zenodo_upload.zip`（约 9MB）整体拖进上传区——Zenodo 把 zip 作为
   归档附件保留（不解包是正常的；可浏览版在 GitHub，Zenodo 这份是永久快照）；
4. 表单逐字段对照 `zenodo/metadata.json` 填写（Description 建议粘
   `zenodo/DESCRIPTION.txt` 纯文本，避免 HTML 标签显示异常）：

| 表单字段 | 填什么 |
|---|---|
| Resource type | Publication → Type: Article |
| Title | `title` 整段 |
| Creators | `Fan, Tianyu`（类型 Individual，Affiliation `Independent researcher`） |
| Description | `zenodo/DESCRIPTION.txt` 纯文本整段 |
| License | 搜选 MIT |
| Keywords | `keywords` 9 个（3D reconstruction, visual hull, identifiability, silhouettes, part assembly, skeleton topology, SDF, solid prior, reproducibility） |
| Version | 1.0.0 |
| Related works | GitHub 仓库 URL，关系 `Is supplement to` |
| Language | English（若有该字段） |

5. 右侧面板 → **DOI** → **Reserve DOI** → 得到 `10.5281/zenodo.XXXXXXX`。
   **先不要点 Publish**，把号码回填（第 3 步）。

## 第 3 步：回填 DOI → 发布 → 推送（5 分钟）

1. DOI 回填两处（`XXXXXXX` 换成真实号码）：
   - `CITATION.cff`：`version: 1.0.0` 下面加一行 `doi: 10.5281/zenodo.XXXXXXX`；
   - `README.md` 的 Cite 段：`See CITATION.cff. DOI: 10.5281/zenodo.XXXXXXX`。
2. 重打 zip（发布的快照自带 DOI）：

```powershell
Compress-Archive -Path E:\Blender\release\* -DestinationPath E:\Blender\zenodo_upload.zip -Force
```

3. Zenodo 草稿页 → 删除旧 zip → 拖入新 zip；
4. 回 GitHub：`git add -A && git commit -m "Add reserved DOI" && git push`；
5. Zenodo 草稿页 → **Publish**（"不可修改"提示属正常；日后要改发 New version，
   concept DOI 不变）；
6. 验收：Zenodo 页面显示 DOI、**Cite this** 按钮（APA/BibTeX 一键复制）、Files 里有 zip；
   GitHub README 的 Cite 段带 DOI。

## 第 4 步（可选）：让别人更容易找到它

- GitHub 仓库 → Releases → 新建 `v1.0.0` release（附 zip；GitHub 与 Zenodo 互通）；
- 把仓库提交到 <https://paperswithcode.com>（有提交入口）；
- `ZONG_REPORT.md` 可作为中文版说明留在仓库。

## 备注

- **arXiv 不作为默认选项**：独立身份首发常卡在 endorsement 制度上；Zenodo DOI 同样可
  引用、免费、无门槛。若以后有合作者能背书，再迁 arXiv 不迟。
- 投稿（AAAI-27 workshop 等）与本存档**互不冲突**：非存档 workshop 均接受已发预印本的稿件。
- 上传失败常见原因：文件名含中文（包内 `总报告.md` 已改名 `ZONG_REPORT.md`）、zip 超 50GB
  （我们仅 9MB，无虞）。
