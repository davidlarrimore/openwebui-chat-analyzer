/**
 * Tests for ConnectionInfoPanel component
 *
 * These tests verify:
 * - Fields are disabled by default and require "Edit Data Source" to enable
 * - Save/Cancel buttons appear only in edit mode
 * - Mode selector and Last Sync render based on sync status
 * - Test Connection shows success/failure states
 * - Load Data triggers correct endpoint
 * - Staleness pill shows correct state
 * - Scheduler drawer functionality
 */

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ConnectionInfoPanel } from "../connection-info-panel";
import * as api from "@/lib/api";

// Mock the API module
jest.mock("@/lib/api");
const mockedApi = api as jest.Mocked<typeof api>;

// Mock toast
jest.mock("@/components/ui/use-toast", () => ({
  toast: jest.fn(),
}));

describe("ConnectionInfoPanel", () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Default mock implementations
    mockedApi.getSyncStatus.mockResolvedValue({
      last_sync_at: "2024-01-01T12:00:00Z",
      last_watermark: "2024-01-01T12:00:00Z",
      has_data: true,
      recommended_mode: "incremental",
      is_stale: false,
      staleness_threshold_hours: 6,
      local_counts: {
        chats: 100,
        messages: 500,
        users: 10,
        models: 5,
      },
    });

    mockedApi.getSyncScheduler.mockResolvedValue({
      enabled: false,
      interval_minutes: 60,
      last_run_at: null,
      next_run_at: null,
    });
  });

  describe("Edit Mode", () => {
    it("should have fields disabled by default", async () => {
      const initialSettings = {
        host: "http://test.com",
        api_key: "test-key",
        database_host: "http://test.com",
        database_api_key: "test-key",
        host_source: "database" as const,
        api_key_source: "database" as const,
      };

      render(<ConnectionInfoPanel initialSettings={initialSettings} />);

      await waitFor(() => {
        const hostnameInput = screen.getByLabelText(/hostname/i);
        const apiKeyInput = screen.getByLabelText(/api key/i);

        expect(hostnameInput).toBeDisabled();
        expect(apiKeyInput).toBeDisabled();
      });
    });

    it("should enable fields when Edit Data Source is clicked", async () => {
      const initialSettings = {
        host: "http://test.com",
        api_key: "test-key",
        database_host: "http://test.com",
        database_api_key: "test-key",
        host_source: "database" as const,
        api_key_source: "database" as const,
      };

      render(<ConnectionInfoPanel initialSettings={initialSettings} />);

      await waitFor(() => screen.getByText(/edit data source/i));

      const editButton = screen.getByText(/edit data source/i);
      fireEvent.click(editButton);

      await waitFor(() => {
        const hostnameInput = screen.getByLabelText(/hostname/i);
        const apiKeyInput = screen.getByLabelText(/api key/i);

        expect(hostnameInput).not.toBeDisabled();
        expect(apiKeyInput).not.toBeDisabled();
      });
    });

    it("should show Save and Cancel buttons in edit mode", async () => {
      const initialSettings = {
        host: "http://test.com",
        api_key: "test-key",
        database_host: "http://test.com",
        database_api_key: "test-key",
        host_source: "database" as const,
        api_key_source: "database" as const,
      };

      render(<ConnectionInfoPanel initialSettings={initialSettings} />);

      // Initially, Save/Cancel should not be visible
      expect(screen.queryByText(/^save settings$/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/^cancel$/i)).not.toBeInTheDocument();

      // Click Edit Data Source
      const editButton = screen.getByText(/edit data source/i);
      fireEvent.click(editButton);

      // Now Save/Cancel should be visible
      await waitFor(() => {
        expect(screen.getByText(/^save settings$/i)).toBeInTheDocument();
        expect(screen.getByText(/^cancel$/i)).toBeInTheDocument();
      });
    });

    it("should call API and exit edit mode when Save is clicked", async () => {
      mockedApi.updateOpenWebUISettings.mockResolvedValue({
        host: "http://new-host.com",
        api_key: "new-key",
        database_host: "http://new-host.com",
        database_api_key: "new-key",
        host_source: "database",
        api_key_source: "database",
      });

      const initialSettings = {
        host: "http://test.com",
        api_key: "test-key",
        database_host: "http://test.com",
        database_api_key: "test-key",
        host_source: "database" as const,
        api_key_source: "database" as const,
      };

      render(<ConnectionInfoPanel initialSettings={initialSettings} />);

      // Enter edit mode
      const editButton = screen.getByText(/edit data source/i);
      fireEvent.click(editButton);

      await waitFor(() => screen.getByLabelText(/hostname/i));

      // Change hostname
      const hostnameInput = screen.getByLabelText(/hostname/i);
      fireEvent.change(hostnameInput, { target: { value: "http://new-host.com" } });

      // Click Save
      const saveButton = screen.getByText(/^save settings$/i);
      fireEvent.click(saveButton);

      await waitFor(() => {
        expect(mockedApi.updateOpenWebUISettings).toHaveBeenCalledWith({
          host: "http://new-host.com",
        });
      });

      // Should exit edit mode (Save/Cancel buttons disappear)
      await waitFor(() => {
        expect(screen.queryByText(/^save settings$/i)).not.toBeInTheDocument();
      });
    });

    it("should revert changes when Cancel is clicked", async () => {
      const initialSettings = {
        host: "http://test.com",
        api_key: "test-key",
        database_host: "http://test.com",
        database_api_key: "test-key",
        host_source: "database" as const,
        api_key_source: "database" as const,
      };

      render(<ConnectionInfoPanel initialSettings={initialSettings} />);

      // Enter edit mode
      const editButton = screen.getByText(/edit data source/i);
      fireEvent.click(editButton);

      await waitFor(() => screen.getByLabelText(/hostname/i));

      // Change hostname
      const hostnameInput = screen.getByLabelText(/hostname/i) as HTMLInputElement;
      fireEvent.change(hostnameInput, { target: { value: "http://changed.com" } });
      expect(hostnameInput.value).toBe("http://changed.com");

      // Click Cancel
      const cancelButton = screen.getByText(/^cancel$/i);
      fireEvent.click(cancelButton);

      // Value should revert
      await waitFor(() => {
        expect(hostnameInput.value).toBe("http://test.com");
      });

      // Should exit edit mode
      expect(screen.queryByText(/^cancel$/i)).not.toBeInTheDocument();
    });
  });

  describe("Sync Status Display", () => {
    it("should display Last Sync timestamp", async () => {
      render(<ConnectionInfoPanel />);

      await waitFor(() => {
        expect(screen.getByText(/last sync:/i)).toBeInTheDocument();
      });
    });

    it("should display staleness pill when data is fresh", async () => {
      mockedApi.getSyncStatus.mockResolvedValue({
        last_sync_at: "2024-01-01T12:00:00Z",
        last_watermark: "2024-01-01T12:00:00Z",
        has_data: true,
        recommended_mode: "incremental",
        is_stale: false,
        staleness_threshold_hours: 6,
        local_counts: { chats: 100, messages: 500, users: 10, models: 5 },
      });

      render(<ConnectionInfoPanel />);

      await waitFor(() => {
        expect(screen.getByText(/up to date/i)).toBeInTheDocument();
      });
    });

    it("should display Stale pill when data is stale", async () => {
      mockedApi.getSyncStatus.mockResolvedValue({
        last_sync_at: "2024-01-01T00:00:00Z",
        last_watermark: "2024-01-01T00:00:00Z",
        has_data: true,
        recommended_mode: "incremental",
        is_stale: true,
        staleness_threshold_hours: 6,
        local_counts: { chats: 100, messages: 500, users: 10, models: 5 },
      });

      render(<ConnectionInfoPanel />);

      await waitFor(() => {
        expect(screen.getByText(/stale/i)).toBeInTheDocument();
      });
    });

    it("should display mode toggle with current mode", async () => {
      render(<ConnectionInfoPanel />);

      await waitFor(() => {
        expect(screen.getByText(/mode: incremental/i)).toBeInTheDocument();
      });
    });
  });

  describe("Test Connection", () => {
    it("should show success toast on successful connection", async () => {
      const { toast } = require("@/components/ui/use-toast");

      mockedApi.testOpenWebUIConnection.mockResolvedValue({
        service: "openwebui",
        status: "ok",
        attempts: 1,
        elapsed_seconds: 0.5,
        meta: {
          version: "0.1.0",
          chat_count: 50,
        },
      });

      render(<ConnectionInfoPanel />);

      await waitFor(() => screen.getByText(/test connection/i));

      const testButton = screen.getByText(/test connection/i);
      fireEvent.click(testButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: "Connection Successful",
            variant: "default",
          })
        );
      });
    });

    it("should show error toast on failed connection", async () => {
      const { toast } = require("@/components/ui/use-toast");

      mockedApi.testOpenWebUIConnection.mockResolvedValue({
        service: "openwebui",
        status: "error",
        attempts: 3,
        elapsed_seconds: 10.0,
        detail: "Connection timeout",
      });

      render(<ConnectionInfoPanel />);

      await waitFor(() => screen.getByText(/test connection/i));

      const testButton = screen.getByText(/test connection/i);
      fireEvent.click(testButton);

      await waitFor(() => {
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: "Connection Failed",
            variant: "destructive",
          })
        );
      });
    });
  });

  describe("Load Data", () => {
    it("should call runSync when Load Data is clicked", async () => {
      mockedApi.runSync.mockResolvedValue({
        detail: "Sync successful",
        dataset: {},
        stats: {},
      });

      const initialSettings = {
        host: "http://test.com",
        api_key: "test-key",
        database_host: "http://test.com",
        database_api_key: "test-key",
        host_source: "database" as const,
        api_key_source: "database" as const,
      };

      render(<ConnectionInfoPanel initialSettings={initialSettings} />);

      await waitFor(() => screen.getByText(/load data/i));

      const loadButton = screen.getByText(/load data/i);
      fireEvent.click(loadButton);

      await waitFor(() => {
        expect(mockedApi.runSync).toHaveBeenCalledWith(
          expect.objectContaining({
            hostname: "http://test.com",
            api_key: "test-key",
            mode: "incremental",
          })
        );
      });
    });
  });

  describe("Scheduler Drawer", () => {
    it("should open scheduler drawer when Scheduler button is clicked", async () => {
      render(<ConnectionInfoPanel />);

      await waitFor(() => screen.getByText(/^scheduler$/i));

      const schedulerButton = screen.getByText(/^scheduler$/i);
      fireEvent.click(schedulerButton);

      await waitFor(() => {
        expect(screen.getByText(/scheduler settings/i)).toBeInTheDocument();
      });
    });

    it("should toggle scheduler when ON/OFF button is clicked", async () => {
      mockedApi.updateSyncScheduler.mockResolvedValue({
        enabled: true,
        interval_minutes: 60,
        last_run_at: null,
        next_run_at: null,
      });

      render(<ConnectionInfoPanel />);

      // Open drawer
      const schedulerButton = screen.getByText(/^scheduler$/i);
      fireEvent.click(schedulerButton);

      await waitFor(() => screen.getByText(/enable scheduler/i));

      // Click toggle
      const toggleButton = screen.getByText(/^off$/i);
      fireEvent.click(toggleButton);

      await waitFor(() => {
        expect(mockedApi.updateSyncScheduler).toHaveBeenCalledWith({
          enabled: true,
        });
      });
    });
  });
});
