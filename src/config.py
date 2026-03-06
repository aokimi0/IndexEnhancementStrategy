"""项目配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ProjectConfig:
    """项目路径和环境变量配置。

    Attributes:
        root_dir: 项目根目录。
        data_dir: 数据目录。
        logs_dir: 日志目录。
        reports_dir: 报告目录。
        cache_dir: 缓存目录。
        tushare_token: Tushare 访问令牌。
    """

    root_dir: Path
    data_dir: Path
    logs_dir: Path
    reports_dir: Path
    cache_dir: Path
    tushare_token: str | None = None

    @classmethod
    def from_root(cls, root_dir: str | Path | None = None) -> "ProjectConfig":
        """从项目根目录构建配置。

        Args:
            root_dir: 项目根目录。为空时自动推断。

        Returns:
            ProjectConfig: 项目配置对象。
        """
        resolved_root = (
            Path(root_dir).resolve()
            if root_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        return cls(
            root_dir=resolved_root,
            data_dir=resolved_root / "data",
            logs_dir=resolved_root / "logs",
            reports_dir=resolved_root / "reports",
            cache_dir=resolved_root / "data" / "cache",
            tushare_token=os.getenv("TUSHARE_TOKEN"),
        )

    def ensure_directories(self) -> None:
        """确保运行所需目录存在。"""
        for path in (
            self.data_dir,
            self.logs_dir,
            self.reports_dir,
            self.cache_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
        ):
            path.mkdir(parents=True, exist_ok=True)
