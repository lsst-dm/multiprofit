# This file is part of multiprofit.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
import astropy.units as u
from collections.abc import Iterable
import lsst.pex.config as pexConfig
import pydantic
from pydantic.dataclasses import dataclass

from .modeller import ModelFitConfig


class CatalogExposureABC(ABC):
    @abstractmethod
    def get_catalog(self) -> Iterable:
        """Return a row-iterable catalog covering an exposure."""


class ColumnInfoConfig:
    arbitrary_types_allowed = True


@dataclass(frozen=True, kw_only=True, config=ColumnInfoConfig)
class ColumnInfo:
    """Metadata for a column in a catalog."""
    dtype: str = pydantic.Field(title="Column data type name (numpy or otherwise)")
    key: str = pydantic.Field(title="Column key (name)")
    description: str = pydantic.Field("", title="Column description")
    unit: u.UnitBase = pydantic.Field(u.Unit(""), title="Column unit (astropy)")


class CatalogFitterConfig(pexConfig.Config):
    """Configuration for generic MultiProFit fitting tasks."""
    column_id = pexConfig.Field[str](default="id", doc="Catalog index column key")
    compute_errors = pexConfig.ChoiceField[str](
        default="INV_HESSIAN_BESTFIT",
        doc="Whether/how to compute sqrt(variances) of each free parameter",
        allowed={
            "NONE": "no errors computed",
            "INV_HESSIAN": "inverse hessian using noisy image as data",
            "INV_HESSIAN_BESTFIT": "inverse hessian using best-fit model as data",
        }
    )
    compute_errors_no_covar = pexConfig.Field[bool](
        default=True,
        doc="Whether to compute parameter errors independently, ignoring covariances",
    )
    config_fit = pexConfig.ConfigField[ModelFitConfig](default=ModelFitConfig(), doc="Fitter configuration")
    fit_centroid = pexConfig.Field[bool](default=True, doc="Fit centroid parameters")
    fit_linear_init = pexConfig.Field[bool](default=True, doc="Fit linear parameters after initialization")
    fit_linear_final = pexConfig.Field[bool](default=True, doc="Fit linear parameters after optimization")
    flag_errors = pexConfig.DictField(default={}, keytype=str, itemtype=str,
                                      doc="Flag column names to set, keyed by name of exception to catch")
    prefix_column = pexConfig.Field[str](default="mpf_", doc="Column name prefix")

    def schema(
        self,
        bands: list[str] = None,
    ) -> list[ColumnInfo]:
        """Return the schema as an ordered list of columns.

        Parameters
        ----------
        bands
            A list of band names to prefix band-dependent columns with.
            Band prefixes should not be used if None.

        Returns
        -------
        An ordered list of ColumnInfo instances.
        """
        schema = [
            ColumnInfo(key=self.column_id, dtype='i8'),
            ColumnInfo(key='n_iter', dtype='i4'),
            ColumnInfo(key='time_eval', dtype='f8', unit=u.s),
            ColumnInfo(key='time_fit', dtype='f8', unit=u.s),
            ColumnInfo(key='time_full', dtype='f8', unit=u.s),
            ColumnInfo(key='chisq_red', dtype='f8'),
            ColumnInfo(key='unknown_flag', dtype='bool'),
        ]
        schema.extend([ColumnInfo(key=key, dtype='bool') for key in self.flag_errors.keys()])
        # Always have a centroid column, even if not fitting
        # It may still be useful for reconstruction
        schema.extend([
            ColumnInfo(key='cen_x', dtype='f8', unit=u.pix),
            ColumnInfo(key='cen_y', dtype='f8', unit=u.pix),
        ])
        return schema
