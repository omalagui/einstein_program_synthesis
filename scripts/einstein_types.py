# einstein_types.py

from __future__ import annotations

class BaseType:
    """
    A base sort.  We use it to represent:
      • Real    (numeric amplitudes, etc.)
      • Move    (moving‐wave sort)
      • Frozen  (frozen‐wave sort)
    and later we build arrow‐types (e.g. (Real -> Real)) on top of it.
    """
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BaseType) and (self.name == other.name)

    def __hash__(self) -> int:
        return hash(self.name)
    
    # ── NEW: intersection operator “&” ─────────────────────────────────────
    def __and__(self, other: "BaseType"):
        """
        Build an IntersectionType when you write  Art & Emp.
        If `other` is not a BaseType, let Python try the reflected op.
        """
        if not isinstance(other, BaseType):
            return NotImplemented
        return IntersectionType(self, other)

    # Support   other & self   when `other`'s class doesn't define __and__
    __rand__ = __and__

class Arrow(BaseType):
    """
    An arrow‐type constructor.  Arrow(src, tgt) represents a function‐type
    from `src` to `tgt`.  For instance, (Real -> Real) is the type of sin, cos, etc.
    """
    def __init__(self, src: BaseType, tgt: BaseType):
        super().__init__(f"({src} -> {tgt})")
        self.src = src
        self.tgt = tgt

    def __repr__(self) -> str:
        return f"({self.src} -> {self.tgt})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Arrow)
            and (self.src == other.src)
            and (self.tgt == other.tgt)
        )

    def __hash__(self) -> int:
        return hash(("arrow", self.src, self.tgt))

# ─── Add This: IntersectionType ─────────────────────────────────────────────────

class IntersectionType(BaseType):
    """
    An intersection type constructor: (left ∧ right).
    Example:
        art = BaseType("art")
        emp = BaseType("emp")
        art_and_emp = IntersectionType(art, emp)   # repr: "(art ∧ emp)"
    """
    def __init__(self, left: BaseType, right: BaseType) -> None:
        # Build the printable name "(left ∧ right)"
        super().__init__(f"({left} ∧ {right})")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, IntersectionType)
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self) -> int:
        return hash(("IntersectionType", self.left, self.right))
    
# ————— Atomic sorts —————
Real   = BaseType("Real")
Move   = BaseType("Move")
Frozen = BaseType("Frozen")
Bot = BaseType("Bot")

Art = BaseType("Art")
Emp = BaseType("Emp")

Art_and_Emp = IntersectionType(Art, Emp)