//! Lock discipline helpers. `DeadlockAsserts` ports Java's `DeadlockAsserts`
//! (spec 02 §5): a `#[cfg(debug_assertions)]` thread-local tracker enforcing the
//! store lock rules. Cheap and worth porting — it caught real bugs in Java.
//!
//! v1 rules tracked here: no store op re-entered from inside a read action
//! (A3 — "must never call back into the store"). The full segment/structural
//! hierarchy rules are added with StoreDirect.

#[cfg(debug_assertions)]
mod imp {
    use std::cell::Cell;

    thread_local! {
        /// Depth of the current push-down action / serializer callback. A store
        /// op invoked while this is non-zero is an A3 violation.
        static IN_ACTION: Cell<u32> = const { Cell::new(0) };
    }

    /// Marks entry into a read action. Returns a guard that decrements on drop.
    pub struct ActionGuard(());

    impl ActionGuard {
        pub fn enter() -> ActionGuard {
            IN_ACTION.with(|c| c.set(c.get() + 1));
            ActionGuard(())
        }
    }
    impl Drop for ActionGuard {
        fn drop(&mut self) {
            IN_ACTION.with(|c| c.set(c.get() - 1));
        }
    }

    /// Assert we are not inside an action (called at the top of every store op).
    pub fn assert_not_in_action(op: &str) {
        IN_ACTION.with(|c| {
            debug_assert!(
                c.get() == 0,
                "store op `{op}` re-entered from inside a read action (A3)"
            );
        });
    }
}

#[cfg(not(debug_assertions))]
mod imp {
    /// No-op guard in release builds.
    pub struct ActionGuard(());
    impl ActionGuard {
        #[inline(always)]
        pub fn enter() -> ActionGuard {
            ActionGuard(())
        }
    }
    #[inline(always)]
    pub fn assert_not_in_action(_op: &str) {}
}

pub use imp::{assert_not_in_action, ActionGuard};
