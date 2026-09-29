# Slide font behavior and notices

`@paatashaala/renderer/fonts.css` defines local-only aliases for six font-family names found in imported slide decks. It loads fonts already installed on the viewer's device. The package does not bundle these font binaries or request them from an external font host; unavailable faces fall back to the browser's normal fonts. The importer leaves the original font-family names in place.

| Slide family | Local font candidates | Original font attribution |
| --- | --- | --- |
| `SourceHanSans` | Source Han Sans, Noto Sans SC | © 2014–2021 Adobe, Reserved Font Name “Source Han Sans”; SIL OFL 1.1 |
| `SourceHanSerif` | Source Han Serif, Noto Serif SC | © 2017–2021 Adobe, Reserved Font Name “Source Han Serif”; SIL OFL 1.1 |
| `LXGWWenKai` | LXGW WenKai | © 2021 The LXGW WenKai Project Authors; SIL OFL 1.1 |
| `ZhuQueFangSong` | Zhuque Fangsong, Noto Serif SC | © 2023 Zhejiang JadeFoci Technology Co. LTD; SIL OFL 1.1 |
| `ZcoolHappy` | ZCOOL KuaiLe | © ZCOOL and the font's designers; [free-use license](font-licenses/ZcoolHappy-LICENSE.txt) |
| `WenDingPLKaiTi` | AR PL KaitiM GB, Noto Serif SC | © 1994–1999 Arphic Technology Co., Ltd.; [Arphic Public License](font-licenses/ARPHIC-PL.txt) |

The full [SIL Open Font License and copyright notices](font-licenses/OFL.txt), [ZCOOL terms](font-licenses/ZcoolHappy-LICENSE.txt), and [Arphic terms](font-licenses/ARPHIC-PL.txt) remain in the repository. Anyone who chooses to redistribute the corresponding font binaries must follow those terms and include the applicable notices. The app's own typography is supplied through its declared font packages, separately from these imported-slide aliases.
