-- Point the cluster at the mounted AGE files (homelab#332, Stage C).
-- Run as the superuser. A reload applies both settings, and ALTER SYSTEM keeps
-- them in the data directory (postgresql.auto.conf), so they survive a restart
-- and a redeploy of the app.
ALTER SYSTEM SET dynamic_library_path = '$libdir:/opt/age/lib';
ALTER SYSTEM SET extension_control_path = '$system:/opt/age/share';
SELECT pg_reload_conf();
