"""Tests for ExitBuilder — declarative exit geometry."""

from src.exits.exit_builder import ExitBuilder, ExitContext, ExitGeometry

TICK_SIZE = 0.25


class TestExitBuilderTargets:
    def test_or_width_target_long(self):
        builder = ExitBuilder(
            target_cfg={"type": "or_width", "multiplier": 1.0},
            stop_cfg={"type": "fixed_ticks", "ticks": 8},
        )
        ctx = ExitContext(entry_price=5600.0, direction="LONG", or_width=4.0)
        geo = builder.compute(ctx)
        assert geo.target_price == 5604.0

    def test_or_width_target_short(self):
        builder = ExitBuilder(
            target_cfg={"type": "or_width", "multiplier": 0.5},
            stop_cfg={"type": "fixed_ticks", "ticks": 8},
        )
        ctx = ExitContext(entry_price=5600.0, direction="SHORT", or_width=4.0)
        geo = builder.compute(ctx)
        assert geo.target_price == 5598.0  # 5600 - 4*0.5

    def test_fixed_ticks_target(self):
        builder = ExitBuilder(
            target_cfg={"type": "fixed_ticks", "ticks": 10},
            stop_cfg={"type": "fixed_ticks", "ticks": 8},
        )
        ctx = ExitContext(entry_price=5600.0, direction="LONG")
        geo = builder.compute(ctx)
        assert geo.target_price == 5600.0 + 10 * TICK_SIZE

    def test_atr_multiple_target(self):
        builder = ExitBuilder(
            target_cfg={"type": "atr_multiple", "multiplier": 2.0},
            stop_cfg={"type": "fixed_ticks", "ticks": 8},
        )
        ctx = ExitContext(entry_price=5600.0, direction="LONG", atr=5.0)
        geo = builder.compute(ctx)
        assert geo.target_price == 5610.0  # 5600 + 5*2

    def test_vwap_target(self):
        builder = ExitBuilder(
            target_cfg={"type": "vwap"},
            stop_cfg={"type": "fixed_ticks", "ticks": 8},
        )
        ctx = ExitContext(entry_price=5595.0, direction="LONG", vwap=5600.0)
        geo = builder.compute(ctx)
        assert geo.target_price == 5600.0


class TestExitBuilderStops:
    def test_first_break_stop_short(self):
        builder = ExitBuilder(
            target_cfg={"type": "fixed_ticks", "ticks": 8},
            stop_cfg={"type": "first_break", "buffer_ticks": 1},
        )
        ctx = ExitContext(
            entry_price=4997.0, direction="SHORT",
            first_break_extreme=5004.0,
        )
        geo = builder.compute(ctx)
        # SHORT stop above extreme: 5004 + 1*0.25 = 5004.25
        assert geo.stop_price == 5004.25

    def test_first_break_stop_long(self):
        builder = ExitBuilder(
            target_cfg={"type": "fixed_ticks", "ticks": 8},
            stop_cfg={"type": "first_break", "buffer_ticks": 2},
        )
        ctx = ExitContext(
            entry_price=5003.0, direction="LONG",
            first_break_extreme=4996.0,
        )
        geo = builder.compute(ctx)
        # LONG stop below extreme: 4996 - 2*0.25 = 4995.5
        assert geo.stop_price == 4995.5

    def test_fixed_ticks_stop(self):
        builder = ExitBuilder(
            target_cfg={"type": "fixed_ticks", "ticks": 8},
            stop_cfg={"type": "fixed_ticks", "ticks": 8},
        )
        ctx = ExitContext(entry_price=5600.0, direction="LONG")
        geo = builder.compute(ctx)
        assert geo.stop_price == 5600.0 - 8 * TICK_SIZE

    def test_atr_multiple_stop(self):
        builder = ExitBuilder(
            target_cfg={"type": "fixed_ticks", "ticks": 8},
            stop_cfg={"type": "atr_multiple", "multiplier": 2.0},
        )
        ctx = ExitContext(entry_price=5600.0, direction="SHORT", atr=5.0)
        geo = builder.compute(ctx)
        assert geo.stop_price == 5610.0  # SHORT stop above: 5600 + 5*2

    def test_or_width_stop(self):
        builder = ExitBuilder(
            target_cfg={"type": "fixed_ticks", "ticks": 8},
            stop_cfg={"type": "or_width", "multiplier": 1.0},
        )
        ctx = ExitContext(entry_price=5600.0, direction="LONG", or_width=4.0)
        geo = builder.compute(ctx)
        assert geo.stop_price == 5596.0  # LONG stop below: 5600 - 4*1.0


class TestExitBuilderFromYaml:
    def test_from_yaml_basic(self):
        exit_cfg = {
            "target": {"type": "or_width", "multiplier": 1.0},
            "stop": {"type": "first_break", "buffer_ticks": 1},
            "time_stop_minutes": 390,
        }
        builder = ExitBuilder.from_yaml(exit_cfg)
        assert builder.time_stop_minutes == 390

        ctx = ExitContext(
            entry_price=5600.0, direction="LONG",
            or_width=4.0, first_break_extreme=5595.0,
        )
        geo = builder.compute(ctx)
        assert geo.target_price == 5604.0
        assert geo.stop_price == 5595.0 - 1 * TICK_SIZE

    def test_from_yaml_slippage_from_orb(self):
        exit_cfg = {"target": {"type": "fixed_ticks", "ticks": 8}}
        orb_cfg = {"slippage_ticks": 2}
        builder = ExitBuilder.from_yaml(exit_cfg, orb_cfg)
        assert builder.slippage_ticks == 2

    def test_from_yaml_defaults(self):
        builder = ExitBuilder.from_yaml({})
        assert builder.time_stop_minutes == 60
        ctx = ExitContext(entry_price=5600.0, direction="LONG")
        geo = builder.compute(ctx)
        # Default: fixed_ticks 8 for both
        assert geo.target_price == 5600.0 + 8 * TICK_SIZE
        assert geo.stop_price == 5600.0 - 8 * TICK_SIZE
